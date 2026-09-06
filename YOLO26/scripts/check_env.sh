#!/bin/bash
# Diagnose why YOLO26's venv can't import torch (ModuleNotFoundError seen on
# the compute node even though run_all.sh's own check should have caught
# that before training started). Run INTERACTIVELY on a GPU node so the CUDA
# check means something:
#
#   qsub -I -q gpu -l select=1:ncpus=2:mem=8gb:ngpus=1 -l walltime=0:30:00
#   bash ~/diplomka/vevees-defects/YOLO26/scripts/check_env.sh
#
# Prints facts only, changes nothing. Paste the full output back for diagnosis.

VENV_DIR="${VENV_DIR:-/storage/brno12-cerit/home/drking/.conda/envs/yolo26}"

section() { printf '\n===== %s =====\n' "$1"; }

section "loaded modules"
module list 2>&1

section "VENV_DIR = $VENV_DIR"
ls -la "$VENV_DIR" 2>&1
echo "--- bin/ ---"
ls -la "$VENV_DIR/bin" 2>&1
echo "--- pyvenv.cfg (shows --system-site-packages target, if any) ---"
cat "$VENV_DIR/pyvenv.cfg" 2>&1
echo "--- bin/python3 resolves to ---"
readlink -f "$VENV_DIR/bin/python3" 2>&1

section "before activation: which python3"
which python3
python3 --version 2>&1

section "activating venv"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
echo "which python: $(which python)"
python --version 2>&1

section "pip list | grep torch/ultralytics"
pip list 2>&1 | grep -iE "torch|ultralytics"
echo "(no lines above means pip sees neither installed in this venv)"

section "actual import test"
python -c "
try:
    import torch
    print('torch OK:', torch.__version__, 'cuda available:', torch.cuda.is_available())
except Exception as e:
    print('torch FAILED:', repr(e))
try:
    import ultralytics
    print('ultralytics OK:', ultralytics.__version__)
except Exception as e:
    print('ultralytics FAILED:', repr(e))
"

section "disk quota (installs can silently fail/truncate over quota)"
quota -s 2>&1
df -h "$VENV_DIR" 2>&1

deactivate 2>/dev/null
echo
echo "Done -- paste everything above back for diagnosis."
