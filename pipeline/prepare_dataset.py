"""Build a YOLO classification dataset from the raw, filename-labeled photos.

Scans RAW_DATA_DIRS for *.jpg files whose stem matches one of
config.INCLUDED_CLASSES, de-duplicates identical files (the raw set has an
exact duplicate between set01/set02), downsizes them, and writes:

    dataset/train/<class>/<name>.jpg
    dataset/val/<class>/<name>.jpg

ready for `yolo classify train data=dataset ...`.

Usage: python -m pipeline.prepare_dataset
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from pipeline.config import (
    DATASET_DIR,
    INCLUDED_CLASSES,
    JPEG_QUALITY,
    MIN_UNIQUE_FOR_VAL_SPLIT,
    RESIZE_MAX_SIDE,
    VAL_HOLDOUT_PER_CLASS,
)
from pipeline.dataset_scan import dedupe, discover

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 0


def resize_and_save(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = RESIZE_MAX_SIDE / max(w, h)
        if scale < 1:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=JPEG_QUALITY)


def main() -> None:
    random.seed(SEED)
    by_class = discover(INCLUDED_CLASSES)

    dataset_root = REPO_ROOT / DATASET_DIR
    if dataset_root.exists():
        for old in dataset_root.rglob("*.jpg"):
            old.unlink()

    print("Class summary:")
    per_class_unique: dict[str, list[Path]] = {}
    for cls, paths in by_class.items():
        print(f"  {cls}: {len(paths)} raw file(s) found")
        per_class_unique[cls] = dedupe(paths)

    # ultralytics classify requires train/ and val/ to contain the exact same
    # set of class folders, so a per-class holdout is only fair -- and only
    # possible -- if every class clears the threshold. Right now several
    # classes have just 1-2 unique images, so there is no way to hold out a
    # sample without either leaving a class empty in train or in val.
    fair_split_possible = all(
        len(u) >= MIN_UNIQUE_FOR_VAL_SPLIT for u in per_class_unique.values()
    )
    if not fair_split_possible:
        print(
            "\nNot enough unique images per class for a fair holdout "
            f"(need >= {MIN_UNIQUE_FOR_VAL_SPLIT} in every class) -- training "
            "on everything and mirroring train into val just so the ultralytics "
            "classify trainer has a val/ split to run against. Treat the "
            "resulting val metrics as a pipeline smoke test, not a real "
            "accuracy estimate."
        )

    total_train = 0
    total_val = 0
    for cls, unique in per_class_unique.items():
        random.shuffle(unique)

        if fair_split_possible:
            val = unique[:VAL_HOLDOUT_PER_CLASS]
            train = unique[VAL_HOLDOUT_PER_CLASS:]
        else:
            val = unique
            train = unique

        for p in train:
            resize_and_save(p, dataset_root / "train" / cls / p.name)
        for p in val:
            resize_and_save(p, dataset_root / "val" / cls / p.name)

        total_train += len(train)
        total_val += len(val)
        print(f"  {cls}: -> train={len(train)} val={len(val)}")

    print(f"\nWrote {total_train} train / {total_val} val images to {dataset_root}")


if __name__ == "__main__":
    main()
