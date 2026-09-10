"""Package the hand-drawn boxes from `python -m pipeline.annotate` into a
YOLO detection dataset:

    detect_dataset/images/train/<name>.jpg
    detect_dataset/labels/train/<name>.txt   (one "class_id cx cy w h" line per box)
    detect_dataset/images/val/...
    detect_dataset/labels/val/...
    detect_dataset/data.yaml

Reads directly from `annotations/boxes.json` rather than assuming one class
per image, since a photo can carry multiple boxes/classes.

Train/val split: 1-3 images/class is too little for a fair per-class
holdout, so if every class clears MIN_UNIQUE_FOR_VAL_SPLIT, hold out one
whole image for val; otherwise mirror train into val, same as
prepare_dataset.py.

Usage: python -m pipeline.prepare_detect_dataset
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

from pipeline.annotate import ANNOTATIONS_PATH, build_image_list, load_annotations
from pipeline.config import (
    DETECT_CLASS_NAMES,
    DETECT_DATASET_DIR,
    JPEG_QUALITY,
    MIN_UNIQUE_FOR_VAL_SPLIT,
    RESIZE_MAX_SIDE,
    VAL_HOLDOUT_PER_CLASS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 0


def resize_and_save(src: Path, dst: Path, max_side: int = RESIZE_MAX_SIDE) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=JPEG_QUALITY)


def write_labels(dst: Path, entries: list[tuple[int, list[float]]]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for class_id, box in entries:
        x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in box)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    dst.write_text("\n".join(lines) + "\n")


def main() -> None:
    random.seed(SEED)
    annotations = load_annotations()
    if not annotations:
        raise SystemExit(
            f"no annotations at {ANNOTATIONS_PATH} -- run `python -m pipeline.annotate` first"
        )

    class_id = {name: i for i, name in enumerate(DETECT_CLASS_NAMES)}
    unknown_classes = set()
    items: list[tuple[str, list[tuple[str, list[float]]]]] = []
    per_class_image_count: dict[str, set[str]] = defaultdict(set)

    for rel_path, boxes in annotations.items():
        entries = []
        for b in boxes:
            if b["cls"] not in class_id:
                unknown_classes.add(b["cls"])
                continue
            entries.append((b["cls"], b["box"]))
            per_class_image_count[b["cls"]].add(rel_path)
        if entries:
            items.append((rel_path, entries))

    if unknown_classes:
        print(f"warning: ignoring boxes with unrecognized class(es): {sorted(unknown_classes)}")

    all_candidates = {it["path"] for it in build_image_list()}
    annotated_paths = {p for p, _ in items}
    missing = sorted(all_candidates - annotated_paths)

    print("Class summary (annotated images / total boxes):")
    for name in DETECT_CLASS_NAMES:
        n_images = len(per_class_image_count.get(name, set()))
        n_boxes = sum(1 for _, entries in items for c, _ in entries if c == name)
        flag = "  <- NO ANNOTATED EXAMPLES YET" if n_images == 0 else ""
        print(f"  {name}: {n_images} images / {n_boxes} boxes{flag}")

    if missing:
        print("\nNot yet annotated (skipped):")
        for m in missing:
            print(f"  {m}")

    dataset_root = REPO_ROOT / DETECT_DATASET_DIR
    if dataset_root.exists():
        for old in dataset_root.rglob("*"):
            if old.is_file():
                old.unlink()

    fair_split_possible = all(
        len(paths) >= MIN_UNIQUE_FOR_VAL_SPLIT for paths in per_class_image_count.values()
    )
    if not fair_split_possible:
        print(
            "\nNot enough annotated images for at least one class (need >= "
            f"{MIN_UNIQUE_FOR_VAL_SPLIT}) -- training on everything and mirroring "
            "train into val, same as the classifier. Treat val metrics as a "
            "pipeline smoke test, not a real accuracy estimate."
        )

    shuffled = items[:]
    random.shuffle(shuffled)
    if fair_split_possible:
        n_val = min(VAL_HOLDOUT_PER_CLASS, max(0, len(shuffled) - 1))
        val_items = shuffled[:n_val]
        train_items = shuffled[n_val:]
    else:
        val_items = shuffled
        train_items = shuffled

    for split, split_items in (("train", train_items), ("val", val_items)):
        for rel_path, entries in split_items:
            p = REPO_ROOT / rel_path
            resize_and_save(p, dataset_root / "images" / split / p.name)
            label_entries = [(class_id[c], box) for c, box in entries]
            write_labels(dataset_root / "labels" / split / (p.stem + ".txt"), label_entries)

    data_yaml = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(DETECT_CLASS_NAMES)},
    }
    (dataset_root / "data.yaml").write_text(yaml.dump(data_yaml, sort_keys=False))

    print(f"\nWrote {len(train_items)} train / {len(val_items)} val images to {dataset_root}")
    print(f"data.yaml: {dataset_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
