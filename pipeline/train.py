"""Fine-tune a pretrained YOLOv8 classification model on dataset/.

Run `python -m pipeline.prepare_dataset` first to (re)build dataset/.

Usage:
    python -m pipeline.train
    python -m pipeline.train --model yolov8s-cls.pt --epochs 100 --imgsz 640
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from pipeline.config import DATASET_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="yolov8n-cls.pt",
        help="pretrained checkpoint to fine-tune (nano is the fastest on CPU)",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--name", default="defects")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = REPO_ROOT / DATASET_DIR
    if not (dataset_dir / "train").exists():
        raise SystemExit(
            f"{dataset_dir} has no train/ split -- run "
            "`python -m pipeline.prepare_dataset` first"
        )

    model = YOLO(args.model)
    model.train(
        data=str(dataset_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        project=str(REPO_ROOT / "runs" / "classify"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
