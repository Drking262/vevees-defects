"""Fine-tune a stock, pretrained Ultralytics YOLO26 detection model on the
same wood-defects dataset used by ../YOLOV8-CDC (prepared once, shared from
there -- see its data.yaml, not duplicated here).

Unlike YOLOV8-CDC (a custom architecture trained from scratch, no pretrained
weights available for it), YOLO26 ships official COCO-pretrained checkpoints,
so this fine-tunes from one (yolo26n.pt / yolo26s.pt, vendored in this
directory) instead of training from random init -- the standard approach,
and a better fit for a ~4k-image dataset.

Thin wrapper around ultralytics' own train() so the actual resource request
(GPU count, walltime, queue) stays in whatever PBS/qsub script wraps this --
this script only knows about the ML side.

Usage:
    # local CPU smoke test (slow)
    python train_yolo26.py --model yolo26n.pt --device cpu --epochs 1

    # real run on a GPU node (e.g. inside a MetaCentrum job)
    python train_yolo26.py --model yolo26s.pt --device 0 --epochs 100 --batch 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = REPO_ROOT.parent / "YOLOV8-CDC" / "wood_defects_dataset" / "data.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--model", default=str(REPO_ROOT / "yolo26s.pt"), help="pretrained checkpoint to fine-tune from")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0", help="'0' for first GPU, '0,1' for two, 'cpu' for CPU")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=30)
    # Must be absolute: ultralytics auto-inserts the task name ("detect")
    # between its global runs_dir setting and a *relative* project path, so
    # a relative "runs/detect" here doubles into ".../runs/detect/runs/detect".
    p.add_argument("--project", default=str(REPO_ROOT / "runs" / "detect"))
    p.add_argument("--name", default="wood_defects_yolo26")
    args = p.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"no dataset at {args.data}\n"
            "This pipeline shares its dataset with YOLOV8-CDC instead of duplicating it. "
            "Run '../YOLOV8-CDC/prepare_wood_defects.py' first (or check out the repo, "
            "which already vendors it)."
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
