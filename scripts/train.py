#!/usr/bin/env python3
"""
Train YOLO segmentation model, copy best weights to checkpoints, optionally validate and export.

Replaces the former Yolo_Train.ipynb workflow. Uses src.config for paths.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def download_roboflow(api_key: str, workspace: str, project: str, version: int):
    """Download dataset from Roboflow; returns path to data.yaml."""
    from roboflow import Roboflow

    os.makedirs(config.DATASETS_DIR, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov11")
    return os.path.join(dataset.location, "data.yaml")


def train(data_yaml: str, epochs: int = 100, imgsz: int = 640, device: str = "0"):
    """Run YOLO segment training. Returns path to runs directory (e.g. runs/segment/train)."""
    from ultralytics import YOLO

    model = YOLO("yolo11s-seg.pt")
    model.train(
        data=data_yaml,
        task="segment",
        epochs=epochs,
        imgsz=imgsz,
        plots=True,
        device=device,
        project=config.TRAIN_RUNS_DIR,
        name="train",
        exist_ok=True,
    )
    save_dir = getattr(model.trainer, "save_dir", None) if hasattr(model, "trainer") else None
    if not save_dir or not os.path.isdir(save_dir):
        save_dir = os.path.join(config.TRAIN_RUNS_DIR, "train")
    return save_dir


def copy_best_to_checkpoints(runs_dir: str) -> str:
    """Copy runs_dir/weights/best.pt to checkpoints/canopy11s640.pt. Returns destination path."""
    best_pt = Path(runs_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training did not produce {best_pt}")
    shutil.copy(best_pt, config.MODEL_CANOPY)
    print(f"Copied {best_pt} -> {config.MODEL_CANOPY}")
    return config.MODEL_CANOPY


def run_val(model_path: str, data_yaml: str, device: str = "0"):
    """Run validation."""
    from ultralytics import YOLO

    model = YOLO(model_path, task="segment")
    model.val(data=data_yaml, device=device)


def run_predict(model_path: str, source: str, conf: float = 0.15, save: bool = True):
    """Run prediction on a directory or video."""
    from ultralytics import YOLO

    model = YOLO(model_path, task="segment")
    model.predict(source=source, conf=conf, save=save)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO segment model and copy best to checkpoints")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (use if dataset already exists)")
    parser.add_argument("--roboflow-key", type=str, default=None, help="Roboflow API key (to download dataset)")
    parser.add_argument("--workspace", type=str, default="ceeri-dwft0", help="Roboflow workspace")
    parser.add_argument("--project", type=str, default="instance-lzpga", help="Roboflow project")
    parser.add_argument("--version", type=int, default=3, help="Roboflow dataset version")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device (e.g. 0 or cuda:0)")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy best.pt to checkpoints")
    parser.add_argument("--val", action="store_true", help="Run validation after training")
    parser.add_argument("--predict", action="store_true", help="Run predict on Dataset Canopy after training")
    parser.add_argument("--export", action="store_true", help="Export to TensorRT after copying to checkpoints")
    args = parser.parse_args()

    if args.data:
        data_yaml = os.path.abspath(args.data)
        if not os.path.isfile(data_yaml):
            print(f"Error: --data file not found: {data_yaml}", file=sys.stderr)
            sys.exit(1)
    elif args.roboflow_key:
        data_yaml = download_roboflow(args.roboflow_key, args.workspace, args.project, args.version)
    else:
        # Default: look for datasets/Instance-4/data.yaml
        data_yaml = os.path.join(config.DATASETS_DIR, "Instance-4", "data.yaml")
        if not os.path.isfile(data_yaml):
            print(
                "Error: No --data path and no Roboflow --roboflow-key. "
                f"Either pass --data <path/to/data.yaml> or --roboflow-key <key>.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Training with data: {data_yaml}")
    runs_dir = train(data_yaml, epochs=args.epochs, imgsz=args.imgsz, device=args.device)

    model_path = config.MODEL_CANOPY
    if not args.no_copy:
        copy_best_to_checkpoints(runs_dir)
    else:
        model_path = os.path.join(runs_dir, "weights", "best.pt")

    if args.val:
        run_val(model_path, data_yaml, device=args.device)

    if args.predict:
        run_predict(model_path, config.DATASET_DIR, conf=0.15, save=True)

    if args.export:
        from ultralytics import YOLO

        model = YOLO(model_path, task="segment")
        model.export(format="engine", imgsz=640, dynamic=True, batch=16, half=True, device=args.device)
        print("TensorRT export done.")

    print("Training pipeline finished.")


if __name__ == "__main__":
    main()
