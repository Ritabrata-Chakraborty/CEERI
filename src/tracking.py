"""Kalman filter and bbox tracking."""

import cv2
import numpy as np


def _box_iou(box_a, box_b):
    """Compute IoU of two axis-aligned boxes [x1, y1, x2, y2]. Returns 0 if no overlap."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _box_iou_vectorized(box_ref, boxes):
    """IoU of one box_ref [4] with each of boxes (N,4). Returns (N,) float array."""
    ref = np.asarray(box_ref, dtype=np.float64)
    det = np.asarray(boxes, dtype=np.float64)
    if det.size == 0:
        return np.array([], dtype=np.float64)
    if det.ndim == 1:
        det = det.reshape(1, -1)
    x1 = np.maximum(ref[0], det[:, 0])
    y1 = np.maximum(ref[1], det[:, 1])
    x2 = np.minimum(ref[2], det[:, 2])
    y2 = np.minimum(ref[3], det[:, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_ref = (ref[2] - ref[0]) * (ref[3] - ref[1])
    area_det = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    union = area_ref + area_det - inter_area
    return np.where(union > 0, inter_area / union, 0.0)


def init_kalman():
    """Initialize 4-state Kalman filter for bbox tracking."""
    kf = cv2.KalmanFilter(4, 4)
    kf.measurementMatrix = np.eye(4, dtype=np.float32)
    kf.transitionMatrix = np.eye(4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
    return kf


def track_bbox(kalman, selected_bbox, selected_mask, frame, detections, masks, iou_threshold=0.6, draw=True):
    """Update tracking using Kalman filter and IoU matching. Reject detections with IoU < iou_threshold
    with the previous bbox; among the rest pick the one with highest IoU. Returns (new_bbox, new_mask)
    or (None, None) if no detection meets the threshold (caller should go to LOST)."""
    if len(detections) == 0 or len(masks) == 0:
        return (None, None)
    det_arr = np.asarray(detections, dtype=np.float64)
    if det_arr.ndim == 1:
        det_arr = det_arr.reshape(1, -1)
    ious = _box_iou_vectorized(selected_bbox, det_arr)
    valid_idx = np.where(ious >= iou_threshold)[0]
    if len(valid_idx) == 0:
        return (None, None)
    idx = valid_idx[np.argmax(ious[valid_idx])]
    new_bbox = detections[idx]
    new_mask = masks[idx]

    x1, y1, x2, y2 = map(int, new_bbox)
    measured = np.array([[x1], [y1], [x2], [y2]], dtype=np.float32)
    predicted = kalman.predict()
    kalman.correct(measured * 0.15 + predicted * 0.85)

    if draw:
        px1, py1, px2, py2 = map(int, predicted)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
        cv2.putText(frame, "Kalman", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return (new_bbox, new_mask)
