"""Post-processing: NMS, geometry, B-spline and canopy mask processing."""

import numpy as np
import cv2
from numba import jit
from scipy.interpolate import splprep, splev


def prepare_boxes(boxes, scores, labels, masks):
    """Adjust box coordinates to [0, 1], fix ordering, remove zero-area boxes."""
    result_boxes = boxes.copy()
    cond = (result_boxes < 0)
    if cond.any():
        result_boxes[cond] = 0
    cond = (result_boxes > 1)
    if cond.any():
        result_boxes[cond] = 1

    boxes1 = result_boxes.copy()
    result_boxes[:, 0] = np.min(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 2] = np.max(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 1] = np.min(boxes1[:, [1, 3]], axis=1)
    result_boxes[:, 3] = np.max(boxes1[:, [1, 3]], axis=1)

    area = (result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1])
    valid = area > 0
    if not valid.all():
        result_boxes = result_boxes[valid]
        scores = scores[valid]
        labels = labels[valid]
        masks = [m for i, m in enumerate(masks) if valid[i]]

    return result_boxes, scores, labels, masks


def cpu_soft_nms_float(dets, sc, Nt, sigma, thresh, method):
    """Soft-NMS for boxes in [0, 1]."""
    N = dets.shape[0]
    indexes = np.arange(N).reshape(-1, 1)
    dets = np.concatenate((dets, indexes), axis=1)
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores = sc.copy()
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1
        maxpos = pos + np.argmax(scores[pos:]) if pos < N else i
        maxscore = scores[maxpos] if pos < N else scores[-1]

        if tscore < maxscore:
            dets[i], dets[maxpos] = dets[maxpos].copy(), tBD
            scores[i], scores[maxpos] = scores[maxpos], tscore
            areas[i], areas[maxpos] = areas[maxpos], tarea
            tBD, tscore, tarea = dets[i].copy(), scores[i], areas[i]

        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[pos:] - inter + 1e-8)

        if method == 1:
            weight = np.where(ovr > Nt, 1 - ovr, 1.0)
        elif method == 2:
            weight = np.exp(-(ovr ** 2) / sigma)
        else:
            weight = np.where(ovr <= Nt, 1.0, 0.0)
        scores[pos:] *= weight

    keep = dets[:, 4][scores > thresh].astype(int)
    return keep


@jit(nopython=True)
def nms_float_fast(dets, scores, thresh):
    """Fast NMS for boxes in [0, 1]."""
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def nms_method(boxes, scores, labels, masks, method=3, iou_thr=0.5, sigma=0.5, thresh=0.001, weights=None):
    """Run NMS or Soft-NMS per label and return filtered detections with masks."""
    if weights is not None and len(boxes) == len(weights):
        w = np.array(weights)
        for i in range(len(w)):
            scores[i] = np.array(scores[i]) * w[i] / w.sum()

    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    masks = [m for lst in masks for m in lst]
    boxes, scores, labels, masks = prepare_boxes(boxes, scores, labels, masks)

    final_boxes, final_scores, final_labels, final_masks = [], [], [], []
    for label in np.unique(labels):
        cond = labels == label
        bb = boxes[cond]
        ss = scores[cond]
        ll = np.full(np.sum(cond), label)
        mm = [m for i, m in enumerate(masks) if cond[i]]

        if method != 3:
            keep = cpu_soft_nms_float(bb.copy(), ss.copy(), Nt=iou_thr, sigma=sigma, thresh=thresh, method=method)
        else:
            keep = np.array(nms_float_fast(bb, ss, thresh=iou_thr))

        final_boxes.append(bb[keep])
        final_scores.append(ss[keep])
        final_labels.append(ll[keep])
        final_masks.append([mm[i] for i in keep])

    return (
        np.concatenate(final_boxes), np.concatenate(final_scores),
        np.concatenate(final_labels), [m for lst in final_masks for m in lst]
    )


def soft_nms(boxes, scores, labels, masks, method=2, iou_thr=0.5, sigma=0.5, thresh=0.001, weights=None):
    """Soft-NMS wrapper."""
    return nms_method(boxes, scores, labels, masks, method=method, iou_thr=iou_thr, sigma=sigma, thresh=thresh, weights=weights)


def nms_soft(boxes_list, scores_list, labels_list, masks_list):
    """Apply Soft-NMS and return filtered boxes, scores, labels, masks."""
    if not boxes_list:
        return [], [], [], []
    try:
        return soft_nms(boxes_list, scores_list, labels_list, masks_list,
                        iou_thr=0.5, sigma=0.5, thresh=0.5, method=2)
    except ValueError:
        return [], [], [], []


def exponential_smoothing(new_value, prev_value, alpha):
    """Exponential moving average."""
    return alpha * new_value + (1 - alpha) * prev_value


def find_center(mask):
    """Extract row-wise center points. Returns empty array if fewer than 4 points."""
    height = mask.shape[0]
    centers = [
        (int(np.mean(np.where(mask[y, :] > 0)[0])), y)
        for y in range(height)
        if np.any(mask[y, :] > 0)
    ]
    return np.array(centers, dtype=np.float32) if len(centers) >= 4 else np.array([], dtype=np.float32).reshape(0, 2)


def overlay_bspline_on_mask(centers, mask, bspline_smooth, frame_count, k=3, color=(255, 0, 0), alpha=0.35, smoothness=1500, line_thickness=1):
    """Fit B-spline to centers, overlay on mask, return angle and x-offset."""
    angle, x_offset = 0.0, 0.0
    if len(centers) < k + 1:
        return (cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask.copy(), angle, x_offset)

    h, w = mask.shape[:2]
    out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask.copy()

    k_use = min(k, len(centers) - 1)
    tck, _ = splprep([centers[:, 0], centers[:, 1]], k=k_use, s=smoothness)
    u_fine = np.linspace(0, 1, 500)
    x_smooth, y_smooth = splev(u_fine, tck)
    pts = np.column_stack((x_smooth, y_smooth)).astype(np.float32)

    if frame_count == 0 or len(bspline_smooth) == 0:
        bspline_smooth.clear()
        bspline_smooth.extend(pts.tolist())
    else:
        arr = np.array(bspline_smooth, dtype=np.float32)
        arr[:, 0] = exponential_smoothing(x_smooth, arr[:, 0], alpha)
        arr[:, 1] = exponential_smoothing(y_smooth, arr[:, 1], alpha)
        bspline_smooth.clear()
        bspline_smooth.extend(arr.tolist())

    arr = np.array(bspline_smooth, dtype=np.float32)
    xi = np.clip(np.round(arr[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(arr[:, 1]).astype(int), 0, h - 1)
    if len(xi) == 0 or len(yi) == 0:
        return out, angle, x_offset
    for i in range(len(xi) - 1):
        cv2.line(out, (xi[i], yi[i]), (xi[i + 1], yi[i + 1]), color, line_thickness)

    cy = h // 2
    ic = np.argmin(np.abs(yi - cy))
    pt_center = (xi[ic], yi[ic])
    dx, dy = splev(u_fine[ic], tck, der=1)
    angle = np.arctan(dx / (dy + 1e-8))
    x_offset = pt_center[0] - w // 2

    L = 20
    cv2.line(out, (int(pt_center[0] - L * dx), int(pt_center[1] - L * dy)),
             (int(pt_center[0] + L * dx), int(pt_center[1] + L * dy)), color, line_thickness)
    return out, angle, x_offset


def overlay_line_on_mask(points, final_mask, line, frame_count, alpha=0.35, color=(0, 255, 0), thickness=1):
    """Fit line to points, overlay on mask. Returns mask, angle, x_offset."""
    angle, x_offset = 0.0, 0.0
    if len(points) >= 2:
        vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        eps = 1e-6
        m, c = vy / (vx + eps), y - x * vy / (vx + eps)
        topx = int(-c / m)
        bottomx = int((final_mask.shape[0] - c) / (m + eps))

        if frame_count == 0:
            line[:] = np.array([topx, bottomx], dtype=np.float32)
        else:
            line[:] = exponential_smoothing(np.array([topx, bottomx], dtype=np.float32), line, alpha)

        out = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR) if len(final_mask.shape) == 2 else final_mask.copy()
        cv2.line(out, (int(line[0]), 0), (int(line[1]), final_mask.shape[0] - 1), color, thickness)
        angle = np.arctan(1 / m)
        x_offset = (line[0] + line[1] - final_mask.shape[1]) / 2
        return out, angle, x_offset
    out = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR) if len(final_mask.shape) == 2 else final_mask.copy()
    return out, angle, x_offset


def process_canopy_mask(line, center_line, bspline, canopy_mask, frame_count, control_only=False):
    """Process canopy mask: morphology, contours, B-spline and line overlays.
    If control_only=True, only morphology, contours, find_center, and center-line fit are done;
    returns (None, None, None, center_line_mask, 0,0,0,0, angle_center_line, xoffset_center_line) to save copies on edge."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(canopy_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(canopy_mask)
    if contours:
        cv2.drawContours(final_mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)

    centers = find_center(final_mask)
    empty_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR) if len(final_mask.shape) == 2 else final_mask.copy()
    dflt = 0.0
    if len(centers) < 4:
        return (empty_bgr,) * 4 + (dflt,) * 6

    if control_only:
        pts_center = centers.reshape(-1, 1, 2).astype(np.float32)
        center_line_mask, angle_center_line, xoffset_center_line = overlay_line_on_mask(
            pts_center, final_mask, center_line, frame_count, color=(0, 0, 255)
        )
        return (
            None, None, None, center_line_mask,
            dflt, dflt, dflt, dflt,
            angle_center_line, xoffset_center_line
        )

    bspline3_mask, angle_bspline, xoffset_bspline = overlay_bspline_on_mask(
        centers, final_mask.copy(), bspline, frame_count, k=3, color=(0, 0, 255)
    )
    bspline2_mask, angle_line, xoffset_line = overlay_bspline_on_mask(
        centers, final_mask.copy(), bspline, frame_count, k=2, color=(0, 0, 255)
    )
    points = np.column_stack(np.where(final_mask > 0))[:, ::-1]
    line_mask, _, _ = overlay_line_on_mask(points, final_mask.copy(), line, frame_count, color=(0, 0, 255))
    pts_center = centers.reshape(-1, 1, 2).astype(np.float32)
    center_line_mask, angle_center_line, xoffset_center_line = overlay_line_on_mask(
        pts_center, final_mask.copy(), center_line, frame_count, color=(0, 0, 255)
    )

    return (
        bspline3_mask, bspline2_mask, line_mask, center_line_mask,
        angle_bspline, xoffset_bspline, angle_line, xoffset_line,
        angle_center_line, xoffset_center_line
    )
