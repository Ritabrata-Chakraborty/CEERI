"""Allow running as python -m src (same as scripts/run_inference.py)."""

import argparse
import sys
import cv2
from ultralytics import YOLO

from src import config
from src.inference import run_yolo, run_yolo_ui


def main():
    parser = argparse.ArgumentParser(description="Tree line follower inference")
    parser.add_argument("--video", type=str, default=config.DEFAULT_VIDEO,
                        help="Video path or camera index (e.g. 0)")
    parser.add_argument("--model", type=str, default=config.MODEL_CANOPY, help="Model .pt path")
    parser.add_argument("--class", dest="class_id", type=int, default=1,
                        help="Class ID to detect (e.g. 0 or 1). Check model.names for options.")
    parser.add_argument("--ui", action="store_true",
                        help="Use OpenCV UI: click segment to follow, Stop/Redetect controls.")
    args = parser.parse_args()

    model = YOLO(args.model, task="segment")
    names = getattr(model, "names", None) or {}
    if isinstance(names, (list, tuple)):
        valid_class = 0 <= args.class_id < len(names)
    else:
        valid_class = args.class_id in names or len(names) == 0
    if not valid_class:
        print(f"Available classes: {names}", file=sys.stderr)
        sys.exit(1)

    video_src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"Failed to open video/camera: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.ui:
        run_yolo_ui(cap, model, class_id=args.class_id)
    else:
        run_yolo(cap, model)


if __name__ == "__main__":
    main()
