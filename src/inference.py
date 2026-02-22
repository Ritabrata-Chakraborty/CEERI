"""Main inference loop: YOLO + NMS + Kalman tracking + canopy processing."""

import time
import enum
import cv2
import numpy as np

from src import config
from src.tracking import init_kalman, track_bbox
from src.control import control_drone
from src.post_processing import process_canopy_mask, nms_soft

IMG_WIDTH = config.IMG_WIDTH
IMG_HEIGHT = config.IMG_HEIGHT

# Segment display colors: opaque whitish tints (no bboxes). First used for single-segment (tracking).
_SEGMENT_COLORS_BGR = [
    (255, 250, 245),  # slight blue
    (250, 255, 248),  # slight green
    (248, 250, 255),  # slight red
    (250, 248, 255),  # slight magenta
]


class UIState(enum.Enum):
    """UI state: no selection yet, tracking selected segment, lost, or user stopped output."""
    NO_SELECTION = 0   # Waiting for user to click a segment
    TRACKING = 1       # Following selected segment, output enabled
    LOST = 2           # Lost segment, no output, wait for new detection + click
    STOPPED = 3        # User pressed Stop; no output, selection kept


def _draw_fitted_line(frame, angle_rad, x_offset, color=(0, 0, 0), thickness=2):
    """Draw the fitted center line on frame (dark black by default). angle in radians, x_offset in pixels."""
    h, w = frame.shape[:2]
    cx = w / 2.0 + x_offset
    cy = h / 2.0
    # Direction along line: (dx, dy) with angle = arctan(dx/dy)
    dx = np.tan(angle_rad)
    dy = 1.0
    n = np.sqrt(dx * dx + dy * dy)
    if n < 1e-6:
        return
    dx, dy = dx / n, dy / n
    # Extend to top (y=0) and bottom (y=h-1)
    if abs(dy) < 1e-6:
        pt1 = (int(cx), 0)
        pt2 = (int(cx), h - 1)
    else:
        t_top = (0 - cy) / dy
        t_bot = (h - 1 - cy) / dy
        x1 = cx + t_top * dx
        x2 = cx + t_bot * dx
        pt1 = (int(np.clip(x1, 0, w - 1)), 0)
        pt2 = (int(np.clip(x2, 0, w - 1)), h - 1)
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def _segment_contains_point(px, py, boxes_px, masks, w, h):
    """Return index of first segment that contains (px, py), or None. boxes_px in pixel coords."""
    for i, (box, mask) in enumerate(zip(boxes_px, masks)):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            continue
        mh, mw = mask.shape[:2]
        mx = int(px * mw / w) if w else 0
        my = int(py * mh / h) if h else 0
        mx = min(max(mx, 0), mw - 1)
        my = min(max(my, 0), mh - 1)
        if mask[my, mx] > 0:
            return i
    return None


def select_bbox(detections):
    """Display detection centers and prompt user to select one."""
    centers = {i + 1: (int((d[0] + d[2]) / 2), int((d[1] + d[3]) / 2)) for i, d in enumerate(detections)}
    for idx, center in centers.items():
        print(f"{idx} - {center}")
    selected_id = int(input("Select bbox ID: "))
    return selected_id


def run_yolo(cap, model):
    """Main inference loop: YOLO + NMS + Kalman tracking + canopy processing."""
    kalman = init_kalman()
    bspline = []
    tree_line = np.zeros(2, dtype=np.float32)
    tree_center_line = np.zeros(2, dtype=np.float32)

    selected_bbox = None
    printed_once = False
    frame_count = 0
    prev_yaw, prev_roll = 0.0, 0.0

    while True:
        start_time = time.time()
        ret, im0 = cap.read()
        if not ret:
            break

        im0 = cv2.resize(im0, (IMG_WIDTH, IMG_HEIGHT))

        yolo_start = time.time()
        results = model.predict(
            im0, iou=0.95, conf=0.05, classes=[1],
            stream=False, device="cuda:0", verbose=False, half=True
        )
        yolo_time = time.time() - yolo_start

        postprocess_start = time.time()
        boxes_list, scores_list, labels_list, masks_list = [], [], [], []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy()
            boxes[:, [0, 2]] /= IMG_WIDTH
            boxes[:, [1, 3]] /= IMG_HEIGHT
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
            masks_list.append(masks)

        num_detections = sum(b.shape[0] for b in boxes_list)

        if num_detections > 0:
            filtered_boxes, filtered_scores, filtered_labels, filtered_masks = nms_soft(
                boxes_list, scores_list, labels_list, masks_list
            )

            for box, score, label, mask in zip(filtered_boxes, filtered_scores, filtered_labels, filtered_masks):
                x_min, y_min, x_max, y_max = (np.array(box) * [IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT]).astype(int)
                cv2.rectangle(im0, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(im0, f"{label}: {score:.2f}", (x_min, max(y_min - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv11 - Initial Detections", annotated_frame)
            cv2.imshow("YOLOv11 + Kalman Filter", im0)

            canopy_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            detections = (np.array(filtered_boxes) * [IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT]).astype(int)

            if not printed_once:
                print("Displaying initial detections... Press any key to proceed with bbox selection.")
                cv2.waitKey(0)
                selected_id = select_bbox(detections)
                selected_bbox = detections[selected_id - 1]
                selected_mask = filtered_masks[selected_id - 1]
                printed_once = True

            new_bbox, new_mask = track_bbox(
                kalman, selected_bbox, selected_mask, im0, detections, filtered_masks,
                iou_threshold=config.TRACK_IOU_THRESHOLD, draw=True
            )
            if new_bbox is not None and new_mask is not None:
                selected_bbox = new_bbox
                selected_mask = new_mask
            if selected_mask is not None and selected_mask.size > 0:
                sm = cv2.resize(selected_mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                canopy_mask[sm > 0] = 1

            (
                bspline3_mask, bspline2_mask, line_mask, center_line_mask,
                angle_bspline, xoffset_bspline, angle_line, xoffset_line,
                angle_center_line, xoffset_center_line
            ) = process_canopy_mask(tree_line, tree_center_line, bspline, canopy_mask, frame_count)

            yaw, roll = control_drone(canopy_mask, angle_center_line, xoffset_center_line, prev_yaw, prev_roll)
        else:
            yaw, roll = prev_yaw, prev_roll

        if num_detections > 0:
            prev_yaw, prev_roll = yaw, roll

            postprocess_time = time.time() - postprocess_start
            total_time = time.time() - start_time
            fps = 1 / total_time

            print(f"YOLOv11 Inference Time: {yolo_time:.4f}s, FPS: {1/yolo_time:.2f}")
            print(f"Post-Processing Time: {postprocess_time:.4f}s")
            print(f"Total Time: {total_time:.4f}s, FPS: {fps:.2f}")

            cv2.putText(im0, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(im0, f"Yaw: {yaw:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(im0, f"Roll: {roll:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            center = (IMG_WIDTH // 2, IMG_HEIGHT // 2)
            for m in [bspline3_mask, bspline2_mask, line_mask, center_line_mask]:
                cv2.circle(m, center, 2, (255, 0, 0), -1)

            cv2.imshow("Canopy Tracking", bspline2_mask)
            cv2.imshow("YOLOv11 + Kalman Filter", im0)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Side panel width (video stays IMG_WIDTH; panel to the right)
PANEL_WIDTH = 220
# Button rects in panel coords (bx, by, bx+bw, by+bh) for Stop and Redetect
_BTN_GAP, _BTN_H = 10, 32
_BTN_W = PANEL_WIDTH - 2 * _BTN_GAP
_BTN_STOP = (_BTN_GAP, _BTN_GAP, _BTN_GAP + _BTN_W, _BTN_GAP + _BTN_H)
_BTN_REDETECT = (_BTN_GAP, _BTN_GAP + _BTN_H + _BTN_GAP, _BTN_GAP + _BTN_W, _BTN_GAP + _BTN_H + _BTN_GAP + _BTN_H)


def _draw_side_panel(
    state, output_enabled, class_id, class_name,
    yaw, roll, fps, has_detections, prompt_msg
):
    """Draw side panel: buttons on top, then all status/messages below. Returns panel BGR image."""
    h = IMG_HEIGHT
    w = PANEL_WIDTH
    panel = np.ones((h, w, 3), dtype=np.uint8) * 42
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (80, 80, 80), 1)

    # Buttons at top of panel (use module constants)
    cv2.rectangle(panel, (_BTN_STOP[0], _BTN_STOP[1]), (_BTN_STOP[2], _BTN_STOP[3]), (0, 100, 255), 2)
    cv2.putText(panel, "Stop (S)", (_BTN_STOP[0] + 18, _BTN_STOP[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    cv2.rectangle(panel, (_BTN_REDETECT[0], _BTN_REDETECT[1]), (_BTN_REDETECT[2], _BTN_REDETECT[3]), (0, 180, 0), 2)
    cv2.putText(panel, "Restart (R)", (_BTN_REDETECT[0] + 14, _BTN_REDETECT[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)

    # All text messages below the buttons
    line_h = 22
    y_start = _BTN_REDETECT[3] + 20
    x_text = 10

    def put_line(y, text, color=(220, 220, 220)):
        cv2.putText(panel, text, (x_text, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return y + line_h

    yy = y_start
    state_str = state.name if state else "?"
    put_line(yy, f"State: {state_str}", (200, 200, 200))
    yy += line_h
    out_str = "OUTPUT ON" if (output_enabled and state == UIState.TRACKING) else "OUTPUT OFF"
    put_line(yy, out_str, (0, 255, 0) if "ON" in out_str else (100, 100, 100))
    yy += line_h
    put_line(yy, f"Class: {class_id} ({class_name})", (180, 180, 255))
    yy += line_h
    put_line(yy, f"FPS: {fps:.1f}", (180, 255, 180))
    yy += line_h

    if state == UIState.TRACKING or state == UIState.STOPPED:
        put_line(yy, f"Yaw: {yaw:.2f}  Roll: {roll:.2f}", (0, 255, 100))
        yy += line_h
    yy += 8

    # Main user message (no detections, select segment, line lost, etc.)
    if prompt_msg:
        # Word-wrap roughly (break at spaces)
        words = prompt_msg.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 <= 28:
                line = line + (" " if line else "") + word
            else:
                if line:
                    put_line(yy, line, (0, 255, 255))
                    yy += line_h
                line = word
        if line:
            put_line(yy, line, (0, 255, 255))
            yy += line_h
    elif not has_detections:
        put_line(yy, "No detections. Wait for segments.", (150, 150, 255))
        yy += line_h
    elif state == UIState.NO_SELECTION:
        put_line(yy, "Click a segment to follow.", (0, 255, 255))
        yy += line_h
    elif state == UIState.TRACKING:
        put_line(yy, "Tracking. Output active.", (0, 255, 0))
    elif state == UIState.STOPPED:
        put_line(yy, "Stopped. Click Restart to continue.", (0, 165, 255))
        yy += line_h
    elif state == UIState.LOST:
        put_line(yy, "Line lost. Click a segment when visible.", (0, 200, 255))

    return panel


def run_yolo_ui(cap, model, class_id=1):
    """
    Run inference with OpenCV UI: always show video/camera; click to select segment for line;
    control output only when tracking and not stopped/lost. Stop / Redetect via keys or buttons.
    """
    kalman = init_kalman()
    bspline = []
    tree_line = np.zeros(2, dtype=np.float32)
    tree_center_line = np.zeros(2, dtype=np.float32)

    state = UIState.NO_SELECTION
    output_enabled = True
    selected_bbox = None
    selected_mask = None
    selected_class_id = None  # when set, we only detect this class (tracking); when None, detect all classes
    frame_count = 0
    prev_yaw, prev_roll = 0.0, 0.0
    yaw, roll = 0.0, 0.0
    fps = 0.0

    class_names = getattr(model, "names", {})
    class_name = class_names.get(class_id, f"class_{class_id}")

    # Shared context for mouse callback (current frame detections)
    ui_ctx = {
        "click_xy": None,
        "filtered_boxes_px": [],
        "filtered_masks": [],
        "filtered_scores": [],
        "filtered_labels": [],
    }

    win = "Tree Line Follower"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def _on_mouse(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ui_ctx["click_xy"] = (x, y)

    cv2.setMouseCallback(win, _on_mouse)

    try:
        while True:
            start_time = time.time()
            ret, im0 = cap.read()
            if not ret:
                break

            im0 = cv2.resize(im0, (IMG_WIDTH, IMG_HEIGHT))
            display = im0.copy()
            prompt_msg = None
            num_detections = 0
            filtered_boxes = []
            filtered_scores = []
            filtered_labels = []
            filtered_masks = []
            detections_px = []

            if state == UIState.STOPPED:
                # No detection/segmentation when stopped: raw feed only
                prompt_msg = "Stopped. Click Restart to continue."
            else:
                # When a segment is selected (TRACKING), detect only that class for Kalman/downstream.
                # When no selection (NO_SELECTION, LOST), detect all classes and show all segments.
                # Explicitly pass all class IDs when no selection so predictor never reuses a previous classes filter.
                predict_kw = dict(
                    iou=0.95, conf=0.05,
                    stream=False, device="cuda:0", verbose=False, half=True
                )
                if selected_class_id is not None:
                    predict_kw["classes"] = [selected_class_id]
                else:
                    all_class_ids = list(class_names.keys()) if isinstance(class_names, dict) else list(range(len(class_names)))
                    if all_class_ids:
                        predict_kw["classes"] = all_class_ids
                results = model.predict(im0, **predict_kw)

                boxes_list, scores_list, labels_list, masks_list = [], [], [], []
                for result in results:
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    boxes = result.boxes.xyxy.cpu().numpy()
                    boxes = (boxes / np.array([IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT], dtype=np.float32)).astype(np.float32)
                    scores = result.boxes.conf.cpu().numpy()
                    labels = result.boxes.cls.cpu().numpy().astype(int)
                    masks = result.masks.data.cpu().numpy()
                    boxes_list.append(boxes)
                    scores_list.append(scores)
                    labels_list.append(labels)
                    masks_list.append(masks)

                num_detections = sum(b.shape[0] for b in boxes_list)

            if num_detections > 0:
                filtered_boxes, filtered_scores, filtered_labels, filtered_masks = nms_soft(
                    boxes_list, scores_list, labels_list, masks_list
                )
                detections_px = (np.array(filtered_boxes) * [IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT]).astype(int)
                ui_ctx["filtered_boxes_px"] = detections_px
                ui_ctx["filtered_masks"] = filtered_masks
                ui_ctx["filtered_scores"] = filtered_scores
                ui_ctx["filtered_labels"] = filtered_labels

                # When TRACKING: show only selected segment + line (drawn below). When STOPPED or no selection: show all segments.
                if state != UIState.TRACKING or selected_mask is None:
                    for i, mask in enumerate(filtered_masks):
                        if mask.size > 0:
                            m_resized = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                            color = _SEGMENT_COLORS_BGR[i % len(_SEGMENT_COLORS_BGR)]
                            display[m_resized > 0] = color

            # Handle click: video area = segment selection; panel area = buttons
            click_xy = ui_ctx.get("click_xy")
            if click_xy is not None:
                ui_ctx["click_xy"] = None
                cx, cy = click_xy
                if cx >= IMG_WIDTH:
                    # Click in side panel (panel coords: px, cy)
                    px = cx - IMG_WIDTH
                    if _BTN_STOP[0] <= px <= _BTN_STOP[2] and _BTN_STOP[1] <= cy <= _BTN_STOP[3]:
                        output_enabled = False
                        state = UIState.STOPPED
                        selected_bbox = None
                        selected_mask = None
                        selected_class_id = None
                    elif _BTN_REDETECT[0] <= px <= _BTN_REDETECT[2] and _BTN_REDETECT[1] <= cy <= _BTN_REDETECT[3]:
                        state = UIState.NO_SELECTION
                        output_enabled = False
                        selected_bbox = None
                        selected_mask = None
                        selected_class_id = None
                elif state in (UIState.NO_SELECTION, UIState.LOST) and num_detections > 0 and cx < IMG_WIDTH:
                    idx = _segment_contains_point(cx, cy, detections_px, filtered_masks, IMG_WIDTH, IMG_HEIGHT)
                    if idx is not None:
                        selected_bbox = detections_px[idx]
                        selected_mask = filtered_masks[idx]
                        selected_class_id = int(filtered_labels[idx])
                        state = UIState.TRACKING
                        output_enabled = True

            # Tracking path: track selected bbox and compute control
            if state in (UIState.TRACKING, UIState.STOPPED) and selected_bbox is not None and selected_mask is not None:
                if num_detections == 0:
                    state = UIState.LOST
                    output_enabled = False
                    selected_bbox = None
                    selected_mask = None
                    selected_class_id = None
                    prompt_msg = "Line lost. Click a segment when visible."
                else:
                    new_bbox, new_mask = track_bbox(
                        kalman, selected_bbox, selected_mask, display,
                        detections_px, filtered_masks,
                        iou_threshold=config.TRACK_IOU_THRESHOLD, draw=False
                    )
                    if new_bbox is None or new_mask is None:
                        state = UIState.LOST
                        output_enabled = False
                        selected_bbox = None
                        selected_mask = None
                        selected_class_id = None
                        prompt_msg = "Line lost. Click a segment when visible."
                    else:
                        selected_bbox = new_bbox
                        selected_mask = new_mask
                        canopy_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
                        if selected_mask.size > 0:
                            sm = cv2.resize(selected_mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                            canopy_mask[sm > 0] = 1
                        (
                            _, _, _, center_line_mask,
                            _, _, _, _,
                            angle_center_line, xoffset_center_line
                        ) = process_canopy_mask(
                            tree_line, tree_center_line, bspline, canopy_mask, frame_count, control_only=True
                        )
                        yaw, roll = control_drone(canopy_mask, angle_center_line, xoffset_center_line, prev_yaw, prev_roll)
                        prev_yaw, prev_roll = yaw, roll
                        # Only output when tracking and output enabled (not stopped/lost)
                        if state == UIState.STOPPED or not output_enabled:
                            yaw, roll = 0.0, 0.0  # no control output
                        # When TRACKING only: show selected segment + line; when STOPPED we already drew all segments above
                        if state == UIState.TRACKING:
                            if selected_mask.size > 0:
                                sm = cv2.resize(selected_mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                                display[sm > 0] = _SEGMENT_COLORS_BGR[0]
                            _draw_fitted_line(display, angle_center_line, xoffset_center_line, color=(0, 0, 0), thickness=2)
            else:
                if state == UIState.LOST:
                    prompt_msg = "Line lost. Click a segment when visible."
                elif state == UIState.NO_SELECTION and num_detections > 0:
                    prompt_msg = "Click a segment to follow."
                elif state == UIState.NO_SELECTION and num_detections == 0:
                    prompt_msg = "No detections. Wait for segments."
                elif state == UIState.TRACKING:
                    prompt_msg = "Tracking. Output active."

            fps = 1.0 / max(time.time() - start_time, 1e-6)
            # Panel class label: show selected class when tracking, else "All"
            panel_class_id = selected_class_id if selected_class_id is not None else class_id
            panel_class_name = class_names.get(panel_class_id, f"class_{panel_class_id}")
            if selected_class_id is None:
                panel_class_name = "All"
            panel = _draw_side_panel(
                state, output_enabled, panel_class_id, panel_class_name,
                yaw, roll, fps, num_detections > 0, prompt_msg
            )
            composite = np.hstack([display, panel])
            cv2.imshow(win, composite)

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                output_enabled = False
                state = UIState.STOPPED
                selected_bbox = None
                selected_mask = None
                selected_class_id = None
            if key == ord("r"):
                state = UIState.NO_SELECTION
                output_enabled = False
                selected_bbox = None
                selected_mask = None
                selected_class_id = None

    finally:
        cap.release()
        cv2.destroyAllWindows()
