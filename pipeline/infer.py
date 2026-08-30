"""Run a trained (or pretrained) YOLO classifier on an image or a folder of images.

Usage:
    python -m pipeline.infer path/to/image_or_folder
    python -m pipeline.infer path/to/image_or_folder --weights runs/classify/defects/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "classify" / "defects" / "weights" / "best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_source(source: str) -> str | list[str]:
    p = Path(source)
    if p.is_dir():
        # Ultralytics' directory loader only looks at the top level, so walk
        # recursively ourselves to also pick up class-subfolder layouts like
        # dataset/val/<class>/*.jpg.
        files = sorted(
            str(f) for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS
        )
        if not files:
            raise SystemExit(f"no images found under {p}")
        return files
    return source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="image file or directory of images")
    p.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="path to a trained .pt checkpoint (default: latest pipeline.train run)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--topk", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"no weights at {weights} -- train first with `python -m pipeline.train`, "
            "or pass --weights"
        )

    model = YOLO(str(weights))
    results = model.predict(source=resolve_source(args.source), imgsz=args.imgsz, device="cpu")

    for r in results:
        names = r.names
        probs = r.probs
        top_idx = probs.top5[: args.topk]
        top_conf = probs.top5conf[: args.topk].tolist()
        print(r.path)
        for idx, conf in zip(top_idx, top_conf):
            print(f"  {names[idx]:<35s} {conf:.3f}")


if __name__ == "__main__":
    main()
