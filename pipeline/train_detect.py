"""Fine-tune a pretrained YOLOv8 *detection* model on detect_dataset/.

Run, in order:
    python -m pipeline.annotate               # draw boxes
    python -m pipeline.prepare_detect_dataset  # package them
    python -m pipeline.train_detect            # this

Usage:
    python -m pipeline.train_detect
    python -m pipeline.train_detect --model yolov8s.pt --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from pipeline.config import DETECT_DATASET_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="yolov8n.pt",
        help="pretrained detection checkpoint to fine-tune (nano is the fastest on CPU)",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--name", default="defects")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = REPO_ROOT / DETECT_DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"{data_yaml} missing -- run `python -m pipeline.annotate` then "
            "`python -m pipeline.prepare_detect_dataset` first"
        )

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        project=str(REPO_ROOT / "runs" / "detect"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
