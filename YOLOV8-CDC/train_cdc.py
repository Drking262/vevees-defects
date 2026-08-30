"""Train yolov8_CDC.yaml (the custom ADown2/C2f_DWRSeg/MSDA-attention
architecture) on a dataset prepared by prepare_wood_defects.py.

Thin wrapper around ultralytics' own train() so the actual resource request
(GPU count, walltime, queue) stays in whatever PBS/qsub script wraps this --
this script only knows about the ML side.

Usage:
    # local CPU smoke test (slow -- a few minutes per epoch on ~4k images)
    python train_cdc.py --data wood_defects_dataset/data.yaml --device cpu --epochs 1

    # real run on a GPU node (e.g. inside a MetaCentrum job)
    python train_cdc.py --data wood_defects_dataset/data.yaml --device 0 --epochs 150 --batch 32
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="wood_defects_dataset/data.yaml")
    p.add_argument("--model", default="yolov8_CDC.yaml")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0", help="'0' for first GPU, '0,1' for two, 'cpu' for CPU")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="wood_defects_cdc")
    args = p.parse_args()

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
