#!/usr/bin/env python3
"""Run the full tree-canopy inference pipeline (YOLO + Kalman + NMS + canopy processing)."""

import argparse
import os
import sys
import threading

import cv2
from ultralytics import YOLO

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.inference import run_yolo, run_yolo_ui

# Substrings in stderr lines that we suppress (Qt font messages)
_QT_FONT_SUPPRESS = ("QFontDatabase", "font directory", "Note that Qt")


def _stderr_filter_worker(pipe_r, orig_fd):
    """Read from pipe_r, write to orig_fd only lines not matching _QT_FONT_SUPPRESS."""
    with os.fdopen(pipe_r, "rb") as f:
        buf = b""
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf or (not chunk and buf):
                line, _, buf = buf.partition(b"\n")
                if not line and not buf:
                    break
                try:
                    text = line.decode("utf-8", errors="replace")
                except Exception:
                    text = str(line)
                if not any(s in text for s in _QT_FONT_SUPPRESS):
                    try:
                        os.write(orig_fd, line + b"\n")
                    except OSError:
                        pass
        if buf and not any(s in buf.decode("utf-8", errors="replace") for s in _QT_FONT_SUPPRESS):
            try:
                os.write(orig_fd, buf)
            except OSError:
                pass


def _run_ui_with_suppressed_qt_stderr(cap, model, class_id):
    """Run UI while filtering Qt font-directory messages from C stderr."""
    if os.environ.get("TREE_LINE_FOLLOWER_SHOW_QT_FONTS", "").lower() in ("1", "true", "yes"):
        run_yolo_ui(cap, model, class_id=class_id)
        return
    stderr_fd = sys.stderr.fileno()
    orig_fd = os.dup(stderr_fd)
    r, w = os.pipe()
    os.dup2(w, stderr_fd)
    os.close(w)  # only stderr_fd (2) is the pipe write end now
    reader = threading.Thread(target=_stderr_filter_worker, args=(r, orig_fd), daemon=True)
    reader.start()
    try:
        run_yolo_ui(cap, model, class_id=class_id)
    finally:
        os.close(stderr_fd)  # close pipe write end so reader gets EOF
        reader.join(timeout=1.0)
        os.dup2(orig_fd, stderr_fd)
        os.close(orig_fd)


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
        print(f"Failed to open video/camera: {args.video}")
        sys.exit(1)

    if args.ui:
        _run_ui_with_suppressed_qt_stderr(cap, model, args.class_id)
    else:
        run_yolo(cap, model)


if __name__ == "__main__":
    main()
