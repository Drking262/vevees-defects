"""Zero-shot localized-defect detection with a pretrained open-vocabulary
model (YOLO-World) -- no wood-specific training at all.

Every other localized-defect tool in this pipeline (pipeline/train.py,
pipeline/train_detect.py) fine-tunes on this project's ~15 photos. This
module instead loads YOLO-World's stock pretrained weights (auto-downloaded
from Ultralytics, trained on large web image-text datasets, never shown a
wood photo) and prompts it with plain-English phrases for the defect classes
via `model.set_classes(...)` -- no training step, no annotation needed.

Usage:
    python -m pipeline.infer_zeroshot path/to/photo.jpg
    python -m pipeline.infer_zeroshot path/to/folder/ --conf 0.01
    python -m pipeline.infer_zeroshot path/to/photo.jpg --classes "knot,crack,hole"

Honest result on this dataset: it doesn't work. Across every included defect
photo, at every image size tried (640 and 1280) and every prompt phrasing
tried (specific like "insect hole in wood" and generic like "hole"/"dark
spot"/"circle"), the highest raw confidence YOLO-World assigns to *any* box
on *any* photo is under 0.03 -- indistinguishable from noise, versus the 0.25
default confidence threshold a real detection normally clears. It very
occasionally fires a box at conf ~0.01-0.03 on an arbitrary patch of grain,
not consistently on the actual defect. The model was pretrained on natural/
web imagery (people, furniture, everyday objects) -- close-up wood grain
texture with a knot or rot patch looks nothing like what it learned "hole" or
"crack" to mean, so it has no real signal to key on here. Kept in the
pipeline as a documented negative result and so it's easy to re-try if a
larger/newer open-vocab checkpoint becomes available -- not wired into
pipeline/analyze.py's combined report since it has nothing reliable to add.
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
