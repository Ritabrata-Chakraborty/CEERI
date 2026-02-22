#!/usr/bin/env python3
"""Export YOLO .pt model to TensorRT engine in checkpoints/."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from src import config


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch YOLO to TensorRT")
    parser.add_argument("--model", type=str, default=config.MODEL_CANOPY, help="Input .pt path")
    parser.add_argument("--half", action="store_true", default=True, help="FP16 (default: True)")
    parser.add_argument("--no-half", action="store_false", dest="half")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    model = YOLO(args.model, task="segment")
    model.export(
        format="engine",
        imgsz=args.imgsz,
        dynamic=True,
        batch=args.batch,
        half=args.half,
        device=args.device,
    )
    print("Export complete. Engine saved next to model path.")


if __name__ == "__main__":
    main()
