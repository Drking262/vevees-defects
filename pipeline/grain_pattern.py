"""Veneer grain/figure-cut classifier + grain-runout/waviness reporting.

`fladrova_dyha`, `polofladrova_dyha`, `rovnoleta_dyha` and
`podelna_neodlupciva_zrcatka` describe the *cut/figure type* of the veneer
(flame, half-flame, straight-grain, mirror-figure) -- a whole-panel texture
property, not a localized blemish, so (like color_anomaly.py) this is kept
out of the YOLO pipeline. Unlike the color defects though, there's no direct
formula for "which figure type is this" -- it has to be learned from
examples. With only 2 unique reference photos per class there's nowhere
near enough data for a trained CNN to generalize, so this uses a simple,
fully transparent nearest-neighbor match instead: extract a texture/
orientation feature vector (pipeline/grain_features.py) for every reference
photo and for the query image, and report which reference(s) it's closest
to. It gets more accurate purely by adding more reference photos to
data/set01 or data/set02 -- no retraining step required.

`zabeh` (grain runout near a streak) and `sval` (grain waviness) are
reported as raw orientation metrics rather than forced into a yes/no flag:
the 2 reference photos available per class disagreed with each other enough
(see the module's calibration run) that a confident threshold isn't
justified yet.

Usage:
    python -m pipeline.grain_pattern path/to/image_or_folder
    python -m pipeline.grain_pattern --calibrate   # dump features for all of data/
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.config import GRAIN_PATTERN_CLASSES
from pipeline.dataset_scan import dedupe, discover
from pipeline.grain_features import GrainFeatures, extract

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Runout/waviness are reported, not thresholded into a flag -- see docstring.
# These are informal reference bands from the module's --calibrate run, shown
# to give the raw numbers context without claiming a validated cutoff.
RUNOUT_NOTE_DEG = 8.0
ORIENTATION_STD_NOTE_DEG = 15.0


@dataclass
class ReferenceImage:
    cls: str
    features: GrainFeatures


def build_reference_db() -> list[ReferenceImage]:
    by_class = discover(GRAIN_PATTERN_CLASSES)
    refs = []
    for cls, paths in by_class.items():
        for p in dedupe(paths, verbose=False):
            refs.append(ReferenceImage(cls=cls, features=extract(str(p))))
    return refs


def classify(path: str, refs: list[ReferenceImage]) -> list[tuple[str, float, str]]:
    """Returns [(class, distance, reference_filename), ...] sorted nearest-first."""
    query = extract(path).vector()
    ranked = []
    for r in refs:
        dist = float(np.linalg.norm(query - r.features.vector()))
        ranked.append((r.cls, dist, Path(r.features.path).name))
    ranked.sort(key=lambda t: t[1])
    return ranked


def report(path: str, refs: list[ReferenceImage], topk: int = 3) -> None:
    feat = extract(path)
    ranked = classify(path, refs)

    print(path)
    print(f"  runout_deg              {feat.runout_deg}  (note if > {RUNOUT_NOTE_DEG}: possible zabeh/grain runout)")
    print(
        f"  orientation_std_deg     {feat.orientation_std_deg}  "
        f"(note if > {ORIENTATION_STD_NOTE_DEG}: locally inconsistent grain direction, e.g. a knot swirl)"
    )
    print(f"  global_coherence        {feat.global_coherence}")
    print(f"  nearest grain-cut types:")
    for cls, dist, name in ranked[:topk]:
        print(f"    {cls:<30s} dist={dist:.3f}  ({name})")


def resolve_images(source: str) -> list[str]:
    p = Path(source)
    if p.is_dir():
        return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    return [source]


def calibrate() -> None:
    files = sorted(glob.glob(str(REPO_ROOT / "data" / "set*" / "*.jpg")))
    print(f"{'file':<55s} {'coh':>6s} {'runout':>7s} {'ori_std':>8s}")
    for f in files:
        feat = extract(f)
        name = os.path.relpath(f, REPO_ROOT)
        print(
            f"{name:<55s} {feat.global_coherence:6.3f} {feat.runout_deg:7.1f} "
            f"{feat.orientation_std_deg:8.1f}"
        )


def evaluate_leave_one_out() -> None:
    """Honest self-check: for each reference photo, classify it using every
    *other* reference photo and see if the nearest neighbor is the true
    class. With 2 photos/class this is a small, noisy estimate -- but it's
    the only validation the current data supports, and it's more honest than
    reporting no accuracy figure at all."""
    refs = build_reference_db()
    correct = 0
    for i, target in enumerate(refs):
        others = refs[:i] + refs[i + 1 :]
        qvec = target.features.vector()
        dists = sorted(
            ((o.cls, float(np.linalg.norm(qvec - o.features.vector()))) for o in others),
            key=lambda t: t[1],
        )
        pred = dists[0][0]
        ok = pred == target.cls
        correct += ok
        status = "OK   " if ok else "WRONG"
        print(f"  {status} true={target.cls:<30s} pred={pred:<30s} ({Path(target.features.path).name})")
    print(f"\n{correct}/{len(refs)} correct (leave-one-out, n={len(refs)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", nargs="?", help="image file or directory")
    p.add_argument("--calibrate", action="store_true", help="dump features for all of data/")
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="leave-one-out accuracy of the classifier over the reference set",
    )
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    if args.calibrate:
        calibrate()
        return
    if args.evaluate:
        evaluate_leave_one_out()
        return
    if not args.source:
        raise SystemExit("pass an image/folder, or --calibrate/--evaluate")

    refs = build_reference_db()
    for f in resolve_images(args.source):
        report(f, refs, topk=args.topk)


if __name__ == "__main__":
    main()
