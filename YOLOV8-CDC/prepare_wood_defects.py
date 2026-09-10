"""Convert the iluvvatar/wood_surface_defects HF parquet shard(s) into a YOLO
detection dataset (images/ + labels/ + data.yaml) for yolov8_CDC.yaml.

Source boxes are already normalized YOLO format (xc, yc, w, h in [0,1]) --
just a label-name -> class-index mapping and file layout, no conversion.

One shard (~4k images, default) matches the Kaggle version and is what this
project's training actually uses. Pass --shards 5 for the full ~20k-image
dataset.

Usage:
    python prepare_wood_defects.py                  # 1 shard (~4k images), default
    python prepare_wood_defects.py --shards 5        # full dataset (~20k images)
    python prepare_wood_defects.py --out-dir /scratch/wood_defects_dataset --shards 5
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
REPO_ID = "iluvvatar/wood_surface_defects"
SHARD_FILES = [
    "data/train-00000-of-00005-aee708e5dcec3620.parquet",
    "data/train-00001-of-00005-2be8dd5a78dccedb.parquet",
    "data/train-00002-of-00005-81cab1be3ef4b976.parquet",
    "data/train-00003-of-00005-46f348d8df59f63c.parquet",
    "data/train-00004-of-00005-cfde1ad8fb3dd17a.parquet",
]

CLASS_NAMES = sorted(
    [
        "Blue_Stain",
        "Crack",
        "Dead_Knot",
        "Knot_missing",
        "Live_Knot",
        "Marrow",
        "Quartzity",
        "knot_with_crack",
        "overgrown",
        "resin",
    ]
)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
VAL_EVERY = 10  # 1 in 10 rows -> val split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", type=int, default=1, choices=range(1, 6), help="how many of the 5 parquet shards to use")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "wood_defects_dataset"))
    args = p.parse_args()

    # Must be absolute: ultralytics resolves a relative data.yaml `path:` against
    # its own global datasets dir (~/.config/Ultralytics/settings.yaml), not cwd.
    out_dir = Path(args.out_dir).resolve()
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_written = {"train": 0, "val": 0}
    n_boxes = {"train": 0, "val": 0}
    row_idx = 0

    for shard_file in SHARD_FILES[: args.shards]:
        print(f"downloading {shard_file} ...")
        local_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=shard_file)
        pf = pq.ParquetFile(local_path)

        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                split = "val" if row_idx % VAL_EVERY == 0 else "train"
                row_idx += 1

                stem = str(row["id"])
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                img.save(out_dir / "images" / split / f"{stem}.jpg", quality=90)

                lines = []
                for obj in row["objects"]:
                    cls_idx = CLASS_TO_IDX[obj["label"]]
                    xc, yc, w, h = obj["bb"]
                    lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                (out_dir / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )

                n_written[split] += 1
                n_boxes[split] += len(lines)

                if row_idx % 1000 == 0:
                    print(f"...{row_idx} images converted")

    data_yaml = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    (out_dir / "data.yaml").write_text(yaml.dump(data_yaml, sort_keys=False))

    print(f"train: {n_written['train']} images, {n_boxes['train']} boxes")
    print(f"val:   {n_written['val']} images, {n_boxes['val']} boxes")
    print(f"classes: {CLASS_NAMES}")
    print(f"-> {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
