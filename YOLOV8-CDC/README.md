# YOLOV8-CDC

Custom YOLOv8 architecture (ADown2 downsampling, C2f_DWRSeg blocks with a
multi-dilation local-attention module, from `yolov8_CDC.yaml`) from
https://github.com/humblefactos1/YOLOV8-CDC, trained on the public
[VSB-TUO wood surface defects dataset](https://huggingface.co/datasets/iluvvatar/wood_surface_defects)
(sawn-timber knots/cracks/resin/etc., ~20k images) and evaluated against
this project's own oak veneer macro photos in `../data/set01`/`set02`.

The original repo's setup instructions target an ultralytics version from
~late 2023 and don't apply cleanly today. `setup_cdc_ultralytics.py`
automates what still matters (see its docstring for what's skipped: the
paper's WIoU/Focaleriou loss enhancement, plus two unrelated torch/numpy
version-skew fixes).

## Why this needs a GPU

CPU-only here. A calibration run (1 epoch, 3650 images, imgsz 512, batch 16)
took **12.5 minutes** on 16 cores -- a real 150+ epoch run isn't realistic.
Verified the whole pipeline (build/train/eval/checkpoint) works correctly on
CPU at small scale; only speed is the blocker. Meant to run for real on a
GPU (e.g. MetaCentrum).

## Dataset is gitignored, not vendored

`wood_defects_dataset/` (converted to YOLO format by `prepare_wood_defects.py`)
is gitignored -- an 866MB dataset doesn't belong in git history. `run_all.sh`
regenerates it: reuses it if `data.yaml` exists (rewriting its `path:`
field), otherwise downloads from Hugging Face and runs
`prepare_wood_defects.py`. `FORCE=1` (or a different `SHARDS`) forces a
re-download.

## MetaCentrum: `scripts/train_metacentrum.sh`

`qsub YOLOV8-CDC/scripts/train_metacentrum.sh` points `run_all.sh`'s
`VENV_DIR` at a persistent `ptcg` conda env on shared storage instead of
building a fresh venv every run -- needs that env to already exist with a
working CUDA torch. See `../YOLO26/README.md` for why that pipeline doesn't
reuse this same env.

## Quickest path: run_all.sh

`./run_all.sh` runs venv/patch/dataset-prep/train/evaluate in one shot.
Auto-detects GPU vs CPU and picks sane defaults (GPU: 1 shard, 150 epochs;
CPU: 1 shard, 1 epoch smoke test). Safe to re-run; skips work already done.

```bash
DEVICE=0 SHARDS=1 EPOCHS=150 BATCH=32 ./run_all.sh
```

`SHARDS=5` trains on the full ~20k-image dataset instead of the ~4k subset.
See its header comment for all env vars.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate

# CPU-only smoke test:
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
# GPU: install a CUDA build of torch matching the node instead -- see
# https://pytorch.org/get-started/locally/

pip install "ultralytics==8.1.0" pyyaml huggingface_hub pyarrow pillow

python setup_cdc_ultralytics.py   # patches the ultralytics install, runs a smoke test
```

Idempotent, safe to re-run. Ends with a 1-epoch smoke test on ultralytics'
built-in `coco8` dataset -- "smoke test OK" means architecture + ultralytics
version + torch/numpy all work together.

## Usage

```bash
# 1. Build the YOLO dataset (--shards 1 default = ~4k images; --shards 5 = full ~20k)
python prepare_wood_defects.py --shards 1 --out-dir wood_defects_dataset

# 2. Train (needs a GPU -- see above).
python train_cdc.py --data wood_defects_dataset/data.yaml --device 0 --epochs 150 --batch 32

# 3. Evaluate on this project's own oak veneer photos (CPU only).
python evaluate_on_veneer.py --weights runs/detect/wood_defects_cdc/weights/best.pt
```

## Honest expectations

Pretraining data is sawn *timber board* photography; this project's photos
are close-up *oak veneer* shots -- some domain gap is expected even with a
full GPU run. Treat `evaluate_on_veneer.py`'s results as "does transfer
happen at all," not a finished detector. Class sets don't fully overlap
either: nothing here for `hniloba` (rot), `vylomeni` (breakout), or insect
damage specifically.
