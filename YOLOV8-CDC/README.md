# YOLOV8-CDC

Custom YOLOv8 architecture (ADown2 downsampling, C2f_DWRSeg blocks with a
multi-dilation local-attention module, from `yolov8_CDC.yaml`) from
https://github.com/humblefactos1/YOLOV8-CDC, trained here on the public
[VSB-TUO wood surface defects dataset](https://huggingface.co/datasets/iluvvatar/wood_surface_defects)
(sawn-timber knots/cracks/resin/etc., ~20k images) and evaluated against
this project's own oak veneer macro photos in `../data/set01` and
`../data/set02` (the vevees-defects `data/` directory -- this is a
subfolder of vevees-defects, not a separate project).

Original repo instructions ("drop these three files into
`ultralytics/nn/Addmodules`, register them in `nn/tasks.py`, replace
`utils/loss.py`/`utils/metrics.py`") target an ultralytics version from
~late 2023 and don't apply cleanly to a current pip install. `setup_cdc_ultralytics.py`
automates the parts that still matter (see its docstring for exactly what's
skipped and why -- short version: the architecture is intact, only the
paper's custom WIoU/Focaleriou loss enhancement is omitted, and two unrelated
torch/numpy version-skew bugs are patched along the way).

## Why this needs a GPU

This machine is CPU-only. A calibration run (1 epoch, 3650 images, imgsz
512, batch 16) took **12.5 minutes** on a 16-core CPU -- a real training run
(150+ epochs on the full ~20k-image dataset) is not realistic here. Verified
the whole pipeline (build, forward, backward, checkpoint save/reload,
validation) works correctly on CPU at small scale; the only blocker is
speed, not correctness. Meant to run for real on a GPU (e.g. MetaCentrum).

## Quickest path: run_all.sh

`./run_all.sh` runs the whole pipeline below in one shot -- venv, patch,
dataset prep, train, evaluate. Auto-detects a GPU (`nvidia-smi`) and picks
sane defaults for it (5 dataset shards, 150 epochs) vs CPU (1 shard, 1
epoch -- a smoke test, not real training). Safe to re-run; skips work
already done. Drop it straight into your own PBS job body:

```bash
DEVICE=0 SHARDS=5 EPOCHS=150 BATCH=32 ./run_all.sh
```

See its header comment for all the env vars it accepts. The rest of this
README explains what each step it calls actually does, for running them
individually.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate

# CPU-only smoke test:
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
# GPU (e.g. on a MetaCentrum GPU node): install a CUDA build of torch instead,
# matching whatever CUDA the node/container provides -- see
# https://pytorch.org/get-started/locally/

pip install "ultralytics==8.1.0" pyyaml huggingface_hub pyarrow pillow

python setup_cdc_ultralytics.py   # patches the ultralytics install, runs a smoke test
```

`setup_cdc_ultralytics.py` is idempotent -- safe to re-run. It ends with a
real 1-epoch train+val smoke test on ultralytics' tiny built-in `coco8`
dataset; if that prints "smoke test OK", the architecture + this ultralytics
version + this torch/numpy combination all work together.

## Usage

```bash
# 1. Build the YOLO dataset from the public wood-defects dataset.
#    --shards 1 (default, ~4k images) for a quick local check;
#    --shards 5 for the full ~20k images (what a real training run should use).
python prepare_wood_defects.py --shards 5 --out-dir wood_defects_dataset

# 2. Train (this is the part that needs a GPU -- see above).
python train_cdc.py --data wood_defects_dataset/data.yaml --device 0 --epochs 150 --batch 32

# 3. Evaluate the trained detector on this project's own oak veneer photos.
python evaluate_on_veneer.py --weights runs/detect/wood_defects_cdc/weights/best.pt
```

Step 3 only needs CPU and the two folders of veneer photos already in
`../data/set01`/`../data/set02` -- run it back on this machine once
`best.pt` comes back from wherever step 2 ran.

## Honest expectations

The pretraining dataset is sawn *timber board* macro photography (industrial
line-scan style); this project's own photos are close-up single-defect shots
of *oak veneer* sheets. Even with a real GPU training run on 20k images, some
domain gap is expected -- treat `evaluate_on_veneer.py`'s results as "does
pretraining on a larger public dataset transfer at all," not as a finished
detector. The two class sets don't fully overlap either: this dataset covers
knots/cracks/resin/etc. on sawn boards, nothing for `hniloba` (rot),
`vylomeni` (breakout), or insect damage specifically.
