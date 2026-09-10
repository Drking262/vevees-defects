"""Zero-shot localized-defect detection with a pretrained open-vocabulary
model (YOLO-World) -- no wood-specific training at all.

Unlike the other localized-defect tools, which fine-tune on this project's
~15 photos, this loads YOLO-World's stock weights and prompts it with
plain-English defect phrases via `model.set_classes(...)` -- no training or
annotation needed.

Usage:
    python -m pipeline.infer_zeroshot path/to/photo.jpg
    python -m pipeline.infer_zeroshot path/to/folder/ --conf 0.01
    python -m pipeline.infer_zeroshot path/to/photo.jpg --classes "knot,crack,hole"

**Doesn't work on this dataset.** Across every photo, image size (640/1280),
and prompt tried, the highest confidence YOLO-World assigns to any box is
under 0.03 -- noise, versus the 0.25 default threshold, and not consistently
on the actual defect. Pretrained on natural/web imagery, so close-up wood
grain doesn't resemble what it learned "hole"/"crack" to mean. Kept as a
documented negative result, not wired into pipeline/analyze.py's report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from pipeline.config import ZEROSHOT_PROMPTS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = "yolov8s-worldv2.pt"  # auto-downloads to the cwd on first use
DEFAULT_OUT_DIR = REPO_ROOT / "zeroshot_out"
DIAGNOSE_CONF = 0.001  # floor used to surface a "best raw candidate" when nothing clears --conf
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="image file or directory")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO-World checkpoint (auto-downloads)")
    p.add_argument(
        "--classes",
        default=",".join(ZEROSHOT_PROMPTS),
        help="comma-separated open-vocabulary prompts to detect",
    )
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.1, help="confidence threshold")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where to save annotated images")
    args = p.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    model = YOLO(args.weights)
    model.set_classes(classes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = resolve_source(args.source)
    sources = [sources] if isinstance(sources, str) else sources

    for f in sources:
        result = model.predict(source=f, imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)[0]
        print(f)
        if len(result.boxes) == 0:
            diag = model.predict(source=f, imgsz=args.imgsz, conf=DIAGNOSE_CONF, device="cpu", verbose=False)[0]
            if len(diag.boxes) == 0:
                print(f"  (no candidates at all, even down to conf={DIAGNOSE_CONF})")
            else:
                best = diag.boxes[int(diag.boxes.conf.argmax())]
                cls_name = diag.names[int(best.cls.item())]
                conf = float(best.conf.item())
                print(
                    f"  (nothing >= conf={args.conf}; best raw candidate below threshold: "
                    f"{cls_name} at {conf:.3f} -- likely noise, not a real detection)"
                )
        else:
            for box in result.boxes:
                cls_name = result.names[int(box.cls.item())]
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"  {cls_name:<25s} conf={conf:.3f}  box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
            out_path = out_dir / Path(f).name
            result.save(filename=str(out_path))
            print(f"  -> annotated image saved to {out_path}")


if __name__ == "__main__":
    main()
