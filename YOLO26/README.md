# YOLO26

Stock, current Ultralytics model (YOLO26, released January 2026 --
https://docs.ultralytics.com/models/yolo26/), fine-tuned from its official
COCO-pretrained checkpoint on the same [VSB-TUO wood surface defects
dataset](https://huggingface.co/datasets/iluvvatar/wood_surface_defects)
subset as `../YOLOV8-CDC`, and evaluated against this project's own oak
veneer macro photos in `../data/set01` and `../data/set02`.

This is the counterpart to `../YOLOV8-CDC`, not a replacement for it: CDC is
a custom architecture (from a specific paper) trained from scratch, with no
pretrained weights available; this pipeline is the current off-the-shelf
Ultralytics model, fine-tuned from its pretrained checkpoint -- the standard
approach, and a better fit for a ~4k-image dataset. Comparing the two shows
whether the paper's custom architecture buys anything over just using
whatever Ultralytics ships today.

## No downloads needed on MetaCentrum

- **Dataset**: shared with `../YOLOV8-CDC/wood_defects_dataset` (images +
  YOLO labels, vendored in the repo there) instead of a second copy here.
  `run_all.sh` only rewrites that dataset's `data.yaml` `path:` field to
  match the current checkout.
- **Pretrained weights**: `yolo26n.pt` (5.5MB, CPU smoke test) and
  `yolo26s.pt` (20MB, the GPU default) are vendored directly in this
  directory -- `ultralytics` finds them locally and never hits the network.
- **Packages** (`pip install torch ultralytics`) still need network the
  first time `run_all.sh` sets up its `.venv` on a new machine -- that part
  isn't vendored. Unlike YOLOV8-CDC, this pipeline is not pinned to an old
  ultralytics version (`pip install -U ultralytics`, currently tested against
  8.4.138) since YOLO26 needs a current release.

## MetaCentrum: `scripts/train_metacentrum.sh`

`qsub YOLO26/scripts/train_metacentrum.sh` submits a GPU job (queue `gpu`,
8 CPU / 32GB / 1 GPU / 12h walltime -- adjust to your allocation) that loads
Mambaforge, activates a **dedicated** persistent venv at
`~/.conda/envs/yolo26` on shared storage (via `run_all.sh`'s `VENV_DIR`),
and runs `run_all.sh`. First submission builds that venv (needs network for
`pip install torch ultralytics`); every later submission reuses it, no
network needed beyond that.

Deliberately a *separate* env from YOLOV8-CDC's `ptcg` env reuse: CDC pins
`ultralytics==8.1.0` for its architecture patch, this pipeline needs a
current release for YOLO26 -- sharing one env between the two jobs would
mean whichever runs last silently changes the other's ultralytics version.

## Quickest path: run_all.sh

`./run_all.sh` runs the whole pipeline -- venv, dataset check, fine-tune,
evaluate. Auto-detects a GPU (`nvidia-smi`) and picks sane defaults for it
(`yolo26s.pt`, 100 epochs) vs CPU (`yolo26n.pt`, 1 epoch -- a smoke test, not
real training). Safe to re-run. Drop it straight into your own PBS job body:

```bash
DEVICE=0 MODEL=yolo26s.pt EPOCHS=100 BATCH=32 ./run_all.sh
```

See its header comment for all the env vars it accepts.

## Usage (running steps individually)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision   # match your node's CUDA build for GPU
pip install -U ultralytics

# 1. Fine-tune (needs a GPU for a real run -- see above).
python train_yolo26.py --model yolo26s.pt --device 0 --epochs 100 --batch 32

# 2. Evaluate the fine-tuned detector on this project's own oak veneer photos.
python evaluate_on_veneer.py --weights runs/detect/wood_defects_yolo26/weights/best.pt
```

`train_yolo26.py` defaults `--data` to `../YOLOV8-CDC/wood_defects_dataset/data.yaml`.
Step 2 only needs CPU and the two folders of veneer photos already in
`../data/set01`/`../data/set02`.

## Honest expectations

Same domain gap caveat as YOLOV8-CDC: pretraining (COCO) and fine-tuning
(sawn-timber board macro photography) are both a mismatch for this project's
oak veneer sheet photos, and the fine-tuning set only covers
knots/cracks/resin/etc., nothing for `hniloba` (rot), `vylomeni` (breakout),
or insect damage specifically.
