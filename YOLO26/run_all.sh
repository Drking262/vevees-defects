#!/usr/bin/env bash
# End-to-end pipeline: activate venv -> fine-tune a pretrained YOLO26
# checkpoint on the (shared, already-vendored) wood-defects dataset ->
# evaluate on this project's own veneer photos. Safe to re-run.
#
# Nothing gets installed here. VENV_DIR must already have torch + ultralytics
# installed manually (see README.md / error message below).
#
#   DEVICE=0 MODEL=yolo26s.pt EPOCHS=100 BATCH=32 IMGSZ=640 ./run_all.sh
#
# Env vars (all optional):
#   VENV_DIR path to your pre-built venv (default: .venv)
#   DEVICE   ultralytics device string: '0', '0,1', or 'cpu' (default: auto)
#   MODEL    pretrained checkpoint to fine-tune from, vendored in this dir (default: yolo26s.pt on GPU, yolo26n.pt on CPU)
#   EPOCHS   training epochs (default: 100 on GPU, 1 on CPU)
#   BATCH    batch size (default: 32 on GPU, 8 on CPU)
#   IMGSZ    training image size (default: 640)

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
    MODEL="${MODEL:-yolo26n.pt}"
    EPOCHS="${EPOCHS:-1}"
    BATCH="${BATCH:-8}"
    log "No GPU detected (or DEVICE=cpu forced) -- running a CPU SMOKE TEST, not real training"
    echo "1 epoch proves the pipeline works; it will not produce a usable detector."
else
    MODEL="${MODEL:-yolo26s.pt}"
    EPOCHS="${EPOCHS:-100}"
    BATCH="${BATCH:-32}"
    log "GPU detected -- device=$DEVICE"
fi
IMGSZ="${IMGSZ:-640}"

echo "DEVICE=$DEVICE  MODEL=$MODEL  EPOCHS=$EPOCHS  BATCH=$BATCH  IMGSZ=$IMGSZ"

if [[ ! -f "$MODEL" ]]; then
    echo "error: $MODEL not vendored in $REPO_ROOT -- see README.md" >&2
    exit 1
fi

# --- 2. activate venv (must already have torch + ultralytics installed) ----
VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "error: no venv at $VENV_DIR -- create it yourself, e.g.:" >&2
    echo "  python3 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install torch torchvision ultralytics" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MISSING="$(python -c "
import importlib
missing = [m for m in ('torch', 'ultralytics') if importlib.util.find_spec(m) is None]
print(' '.join(missing))
")"
if [[ -n "$MISSING" ]]; then
    echo "error: $VENV_DIR is missing: $MISSING -- install manually, e.g.:" >&2
    echo "  source $VENV_DIR/bin/activate && pip install torch torchvision ultralytics" >&2
    exit 1
fi

# ultralytics auto-enables a wandb logging callback whenever the wandb
# package is importable, even though nothing here asks for it -- on a
# login-less compute node (e.g. MetaCentrum) that crashes training with
# "No API key configured". The real switch is ultralytics' own persisted
# SETTINGS["wandb"] flag, not the WANDB_MODE env var (which only mutes
# wandb's own network calls after ultralytics has already decided to use
# it) -- see ../YOLOV8-CDC/run_all.sh for the failure this avoids.
python -c "from ultralytics.utils import SETTINGS; SETTINGS.update({'wandb': False})"

# --- 3. dataset -------------------------------------------------------------
# Shared with YOLOV8-CDC -- vendored there, not re-downloaded or duplicated
# here. Only the absolute `path:` field needs fixing up per checkout.
DATA_YAML="../YOLOV8-CDC/wood_defects_dataset/data.yaml"
if [[ ! -f "$DATA_YAML" ]]; then
    echo "error: no dataset at $DATA_YAML -- run ../YOLOV8-CDC/prepare_wood_defects.py first" >&2
    exit 1
fi
log "Using shared dataset at $DATA_YAML"
python - "$DATA_YAML" <<'PYEOF'
import sys, yaml, pathlib
p = pathlib.Path(sys.argv[1])
d = yaml.safe_load(p.read_text())
d["path"] = str(p.resolve().parent)
p.write_text(yaml.dump(d, sort_keys=False))
PYEOF

# --- 4. train -----------------------------------------------------------
log "Fine-tuning $MODEL"
python train_yolo26.py --data "$DATA_YAML" --model "$MODEL" --device "$DEVICE" --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ"

BEST_WEIGHTS="runs/detect/wood_defects_yolo26/weights/best.pt"
if [[ ! -f "$BEST_WEIGHTS" ]]; then
    # train_yolo26.py auto-increments the run name (wood_defects_yolo262, ...) on repeat runs
    BEST_WEIGHTS="$(ls -t runs/detect/wood_defects_yolo26*/weights/best.pt 2>/dev/null | head -1)"
fi

# --- 5. evaluate on this project's own veneer photos ------------------------
log "Evaluating on ../data/set01 + ../data/set02"
python evaluate_on_veneer.py --weights "$BEST_WEIGHTS"

log "Done. Weights: $BEST_WEIGHTS"
