"""Unified defect report: runs all three tools on one photo (or a folder).

  1. localized defects -- YOLOv8 *detection* (pipeline/train_detect.py) if
     trained (real boxes, from hand-annotated data); otherwise falls back to
     the YOLOv8 *classifier* (pipeline/train.py, whole-image label only,
     optionally with a Grad-CAM heatmap approximating "where")
  2. color_anomaly.py    -> color-uniformity defects (sapwood, discoloration)
  3. grain_pattern.py    -> veneer figure-cut type + runout/waviness metrics

Each tool targets a disjoint set of defect classes (see pipeline/config.py
for the split and why), so this just runs all three and prints one combined
report per image -- it doesn't try to merge or arbitrate between them.

Usage:
    python -m pipeline.analyze path/to/image_or_folder
    python -m pipeline.analyze path/to/image_or_folder --detect-weights runs/detect/defects/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import color_anomaly, grain_pattern
from pipeline.grain_pattern import build_reference_db

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASSIFY_WEIGHTS = REPO_ROOT / "runs" / "classify" / "defects" / "weights" / "best.pt"
DEFAULT_DETECT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "defects" / "weights" / "best.pt"
DEFAULT_DETECT_OUT_DIR = REPO_ROOT / "detect_out"


def resolve_images(source: str) -> list[str]:
    p = Path(source)
    if p.is_dir():
        return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    return [source]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="image file or directory")
    p.add_argument("--classify-weights", default=str(DEFAULT_CLASSIFY_WEIGHTS))
    p.add_argument("--detect-weights", default=str(DEFAULT_DETECT_WEIGHTS))
    p.add_argument("--detect-out-dir", default=str(DEFAULT_DETECT_OUT_DIR))
    p.add_argument("--topk", type=int, default=3)
    p.add_argument(
        "--gradcam",
        action="store_true",
        help="if only the classifier is available, also save a Grad-CAM heatmap "
        "approximating where the prediction came from",
    )
    p.add_argument(
        "--gradcam-dir",
        default=str(REPO_ROOT / "gradcam_out"),
        help="directory to write --gradcam heatmaps into (default: gradcam_out/, kept out of data/)",
    )
    args = p.parse_args()

    detect_weights = Path(args.detect_weights)
    classify_weights = Path(args.classify_weights)
    use_detect = detect_weights.exists()

    detect_model = None
    classify_model = None
    if use_detect:
        from ultralytics import YOLO

        detect_model = YOLO(str(detect_weights))
    elif classify_weights.exists():
        from ultralytics import YOLO

        classify_model = YOLO(str(classify_weights))
    else:
        print(
            f"(no detection weights at {detect_weights} and no classifier weights at "
            f"{classify_weights} -- skipping localized-defect detection)\n"
        )

    grain_refs = build_reference_db()
    files = resolve_images(args.source)

    for f in files:
        print("=" * 70)
        print(f)
        print("=" * 70)

        if detect_model is not None:
            print("\n[localized defects -- YOLO detection, real boxes]")
            result = detect_model.predict(source=f, imgsz=640, device="cpu", verbose=False)[0]
            if len(result.boxes) == 0:
                print("  (no defect detected above the confidence threshold)")
            for box in result.boxes:
                cls_name = result.names[int(box.cls.item())]
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"  {cls_name:<30s} conf={conf:.3f}  box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
            out_dir = Path(args.detect_out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / Path(f).name
            result.save(filename=str(out_path))
            print(f"  -> annotated image saved to {out_path}")

        elif classify_model is not None:
            print("\n[localized defects -- YOLO classification, whole-image label only]")
            result = classify_model.predict(source=f, imgsz=640, device="cpu", verbose=False)[0]
            names = result.names
            for idx, conf in zip(
                result.probs.top5[: args.topk], result.probs.top5conf[: args.topk].tolist()
            ):
                print(f"  {names[idx]:<35s} {conf:.3f}")

            if args.gradcam:
                from pipeline.gradcam import compute_gradcam, overlay_heatmap

                cropped_img, cam, cls_name, conf = compute_gradcam(str(classify_weights), f)
                overlay = overlay_heatmap(cropped_img, cam)
                out_dir = Path(args.gradcam_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / (Path(f).stem + "_gradcam.jpg")
                overlay.save(out_path)
                print(f"  -> heatmap for '{cls_name}' ({conf:.3f}) saved to {out_path}")

        print("\n[color-based anomalies]")
        color_anomaly.print_report(color_anomaly.analyze(f))

        print("\n[grain/figure type]")
        grain_pattern.report(f, grain_refs, topk=args.topk)
        print()


if __name__ == "__main__":
    main()
