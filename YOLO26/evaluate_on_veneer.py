"""Run the trained YOLO26 wood-defect detector (pretrained on COCO, fine-tuned
on the VSB-TUO wood_surface_defects dataset via train_yolo26.py) against this
project's own oak veneer macro photos (../data/set01, ../data/set02), draw
boxes, and print a report.

Same evaluation this project runs for YOLOV8-CDC (../YOLOV8-CDC/evaluate_on_veneer.py)
against a different model, to compare a custom-architecture from-scratch
detector against a stock, pretrained-then-fine-tuned one.

Usage:
    python evaluate_on_veneer.py [--weights runs/detect/wood_defects_yolo26/weights/best.pt] [--conf 0.15]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent
VEVEES_ROOT = REPO_ROOT.parent
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "wood_defects_yolo26" / "weights" / "best.pt"
DEFAULT_SOURCES = [VEVEES_ROOT / "data" / "set01", VEVEES_ROOT / "data" / "set02"]
DEFAULT_OUT_DIR = REPO_ROOT / "veneer_eval_out"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_images(paths: list[Path]) -> list[str]:
    files: list[str] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(str(f) for f in p.glob("*") if f.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            files.append(str(p))
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--source", nargs="*", default=[str(s) for s in DEFAULT_SOURCES])
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"no weights at {weights} -- run train_yolo26.py first")

    model = YOLO(str(weights))
    files = resolve_images([Path(s) for s in args.source])
    if not files:
        raise SystemExit("no images found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_with_detections = 0
    for f in files:
        result = model.predict(source=f, imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)[0]
        print(Path(f).name)
        if len(result.boxes) == 0:
            print("  (no detection above threshold)")
        else:
            n_with_detections += 1
            for box in result.boxes:
                cls_name = result.names[int(box.cls.item())]
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"  {cls_name:<20s} conf={conf:.3f}  box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
            out_path = out_dir / Path(f).name
            result.save(str(out_path))
            print(f"  -> {out_path}")

    print(f"\n{n_with_detections}/{len(files)} photos got at least one detection >= conf={args.conf}")


if __name__ == "__main__":
    main()
