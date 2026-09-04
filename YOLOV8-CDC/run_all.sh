#!/usr/bin/env bash
# End-to-end pipeline: venv -> patch ultralytics -> prepare wood-defects
# dataset -> train yolov8_CDC -> evaluate on this project's own veneer
# photos. Safe to re-run -- each step skips work that's already done unless
# FORCE=1.
#
# Auto-detects a GPU (via `nvidia-smi`) and picks sane defaults for it
# (1 dataset shard -- the ~4k-image subset matching the Kaggle version of
# this dataset, 150 epochs) vs CPU (1 shard, 1 epoch -- a smoke test, not
# real training; see the repo README for why CPU training isn't realistic
# here). Override anything via environment variables, e.g. from inside your
# own PBS script:
#
#   DEVICE=0 SHARDS=1 EPOCHS=150 BATCH=32 IMGSZ=640 ./run_all.sh
#
# Env vars (all optional):
#   VENV_DIR path to the venv to create/reuse (default: .venv). Point this at
#            a pre-built persistent env (e.g. on shared MetaCentrum storage)
#            to skip venv creation and reuse its already-installed torch/CUDA.
#   DEVICE   ultralytics device string: '0', '0,1', or 'cpu' (default: auto)
#   SHARDS   how many of the 5 wood_surface_defects parquet shards, 1-5 (default: 1 -- the ~4k-image Kaggle-sized subset; pass 5 for the full ~20k-image HF dataset)
#   EPOCHS   training epochs (default: 150 on GPU, 1 on CPU)
#   BATCH    batch size (default: 32 on GPU, 8 on CPU)
#   IMGSZ    training image size (default: 640)
#   FORCE    1 = redo dataset prep even if wood_defects_dataset/data.yaml exists (default: 0)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

log() { printf '\n=== %s ===\n' "$1"; }

# --- 1. GPU detection + defaults -------------------------------------------
if [[ -z "${DEVICE:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        DEVICE="0"
    else
        DEVICE="cpu"
    fi
fi

if [[ "$DEVICE" == "cpu" ]]; then
    SHARDS="${SHARDS:-1}"
    EPOCHS="${EPOCHS:-1}"
    BATCH="${BATCH:-8}"
    log "No GPU detected (or DEVICE=cpu forced) -- running a CPU SMOKE TEST, not real training"
    echo "1 epoch on 1 shard proves the pipeline works; it will not produce a usable detector."
    echo "See README.md for why (measured ~12.5 min/epoch on 3650 images on a 16-core CPU)."
else
    SHARDS="${SHARDS:-1}"
    EPOCHS="${EPOCHS:-150}"
    BATCH="${BATCH:-32}"
    log "GPU detected -- device=$DEVICE"
fi
IMGSZ="${IMGSZ:-640}"
FORCE="${FORCE:-0}"

echo "DEVICE=$DEVICE  SHARDS=$SHARDS  EPOCHS=$EPOCHS  BATCH=$BATCH  IMGSZ=$IMGSZ"

# --- 2. venv + deps ----------------------------------------------------------
# VENV_DIR lets a PBS job point this at a pre-built persistent env (with
# torch/CUDA already installed) instead of creating a fresh local .venv.
VENV_DIR="${VENV_DIR:-.venv}"
log "Setting up venv ($VENV_DIR)"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -c "import torch" >/dev/null 2>&1; then
    if [[ "$DEVICE" == "cpu" ]]; then
        pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    else
        pip install torch torchvision  # pulls a CUDA build; match your node's CUDA if this fails
    fi
fi
pip install -q "ultralytics==8.1.0" pyyaml huggingface_hub pyarrow pillow

# --- 3. patch ultralytics for the CDC architecture ---------------------------
log "Patching ultralytics (idempotent)"
python setup_cdc_ultralytics.py

# --- 4. dataset -----------------------------------------------------------
# wood_defects_dataset/ (images + YOLO labels) ships committed in the repo --
# this never re-downloads it on a fresh clone. FORCE=1 forces a re-download
# (e.g. to switch SHARDS), which does need network access.
DATA_YAML="wood_defects_dataset/data.yaml"
if [[ -f "$DATA_YAML" && "$FORCE" != "1" ]]; then
    log "Dataset already present at $DATA_YAML (set FORCE=1 to re-download)"
    # data.yaml's `path:` is an absolute path baked in wherever it was last
    # generated -- rewrite it to this checkout's location so a repo clone
    # onto a different machine (e.g. MetaCentrum) still resolves correctly.
    python - "$DATA_YAML" <<'PYEOF'
import sys, yaml, pathlib
p = pathlib.Path(sys.argv[1])
d = yaml.safe_load(p.read_text())
d["path"] = str(p.resolve().parent)
p.write_text(yaml.dump(d, sort_keys=False))
PYEOF
else
    log "Preparing wood_surface_defects dataset ($SHARDS shard(s)) -- downloads from Hugging Face"
    python prepare_wood_defects.py --shards "$SHARDS" --out-dir wood_defects_dataset
fi

# --- 5. train -----------------------------------------------------------
log "Training yolov8_CDC"
python train_cdc.py --data "$DATA_YAML" --device "$DEVICE" --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ"

BEST_WEIGHTS="runs/detect/wood_defects_cdc/weights/best.pt"
if [[ ! -f "$BEST_WEIGHTS" ]]; then
    # train_cdc.py auto-increments the run name (wood_defects_cdc2, ...) on repeat runs
    BEST_WEIGHTS="$(ls -t runs/detect/wood_defects_cdc*/weights/best.pt 2>/dev/null | head -1)"
fi

# --- 6. evaluate on this project's own veneer photos ------------------------
log "Evaluating on ../data/set01 + ../data/set02"
python evaluate_on_veneer.py --weights "$BEST_WEIGHTS"

log "Done. Weights: $BEST_WEIGHTS"
