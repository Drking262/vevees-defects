# YOLO26

Stock, current Ultralytics model (YOLO26, released January 2026 --
https://docs.ultralytics.com/models/yolo26/), fine-tuned from its official
COCO-pretrained checkpoint on the same [VSB-TUO wood surface defects
dataset](https://huggingface.co/datasets/iluvvatar/wood_surface_defects)
subset as `../YOLOV8-CDC`, evaluated against `../data/set01`/`set02`.

Counterpart to `../YOLOV8-CDC`, not a replacement: CDC is a custom
architecture trained from scratch with no pretrained weights; this pipeline
fine-tunes Ultralytics' current pretrained checkpoint -- the standard
approach, better fit for a ~4k-image dataset. Comparing the two shows
whether the paper's custom architecture buys anything over stock Ultralytics.

## No downloads needed on MetaCentrum

- **Dataset**: shared with `../YOLOV8-CDC/wood_defects_dataset` instead of a
  second copy here. `run_all.sh` only rewrites its `data.yaml` `path:`.
- **Pretrained weights**: `yolo26n.pt` (5.5MB, CPU smoke test) and
  `yolo26s.pt` (20MB, GPU default) are vendored in this directory.
- **Packages** (`pip install torch ultralytics`) still need network on
  first `.venv` setup. Not pinned like YOLOV8-CDC -- needs a current
  ultralytics release (`pip install -U ultralytics`, tested against 8.4.138).

## MetaCentrum: `scripts/train_metacentrum.sh`

`qsub YOLO26/scripts/train_metacentrum.sh` submits a GPU job that activates
a **dedicated** persistent venv at `~/.conda/envs/yolo26` and runs
`run_all.sh`. First submission builds it (needs network); later ones reuse
it.

Deliberately separate from YOLOV8-CDC's `ptcg` env: CDC pins
`ultralytics==8.1.0`, this pipeline needs a current release -- sharing one
env would mean whichever job runs last changes the other's version.

## Quickest path: run_all.sh

`./run_all.sh` runs venv/dataset-check/fine-tune/evaluate in one shot.
Auto-detects GPU vs CPU (`yolo26s.pt`/100 epochs vs `yolo26n.pt`/1-epoch
smoke test). Safe to re-run.

```bash
DEVICE=0 MODEL=yolo26s.pt EPOCHS=100 BATCH=32 ./run_all.sh
```

See its header comment for all env vars.

## Usage (running steps individually)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision   # match your node's CUDA build for GPU
pip install -U ultralytics

# 1. Fine-tune (needs a GPU for a real run).
python train_yolo26.py --model yolo26s.pt --device 0 --epochs 100 --batch 32

# 2. Evaluate on this project's own oak veneer photos.
python evaluate_on_veneer.py --weights runs/detect/wood_defects_yolo26/weights/best.pt
```

`train_yolo26.py` defaults `--data` to
`../YOLOV8-CDC/wood_defects_dataset/data.yaml`. Step 2 only needs CPU.

## Honest expectations

Same domain gap as YOLOV8-CDC: COCO pretraining and sawn-timber fine-tuning
both mismatch oak veneer sheet photos, and the fine-tuning classes don't
cover `hniloba` (rot), `vylomeni` (breakout), or insect damage.
