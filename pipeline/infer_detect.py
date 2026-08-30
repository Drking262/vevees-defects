"""Run the trained YOLO *detection* model and draw real boxes around defects.

Unlike pipeline/infer.py (whole-image classification) or pipeline/gradcam.py
(an approximate attention heatmap), this draws an actual bounding box from a
model trained on hand-annotated box coordinates (pipeline/annotate.py).

Usage:
    python -m pipeline.infer_detect path/to/image_or_folder
    python -m pipeline.infer_detect path/to/image_or_folder --weights runs/detect/defects/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "defects" / "weights" / "best.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "detect_out"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_source(source: str) -> str | list[str]:
    p = Path(source)
    if p.is_dir():
        files = sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise SystemExit(f"no images found under {p}")
        return files
    return source


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="image file or directory")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where to save annotated images")
    args = p.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"no weights at {weights} -- run pipeline.annotate, "
            "pipeline.prepare_detect_dataset, then pipeline.train_detect first"
        )

    model = YOLO(str(weights))
    results = model.predict(
        source=resolve_source(args.source), imgsz=args.imgsz, conf=args.conf, device="cpu"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        print(r.path)
        if len(r.boxes) == 0:
            print("  (no defect detected above the confidence threshold)")
        for box in r.boxes:
            cls_name = r.names[int(box.cls.item())]
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {cls_name:<30s} conf={conf:.3f}  box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

        out_path = out_dir / Path(r.path).name
        r.save(filename=str(out_path))
        print(f"  -> annotated image saved to {out_path}")


if __name__ == "__main__":
    main()
