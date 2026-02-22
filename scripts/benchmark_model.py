#!/usr/bin/env python3
"""Benchmark YOLO model speed/accuracy (e.g. model.benchmark)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from src import config


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO model")
    parser.add_argument("--model", type=str, default=config.MODEL_CANOPY, help="Model .pt path")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="Dataset yaml for benchmark")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 480], help="Image sizes")
    args = parser.parse_args()

    model = YOLO(args.model, task="segment")
    results = model.benchmark(data=args.data, imgsz=args.imgsz)
    print(results)


if __name__ == "__main__":
    main()
