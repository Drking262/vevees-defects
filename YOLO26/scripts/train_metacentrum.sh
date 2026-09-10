#!/bin/bash
#PBS -N vevees_yolo26_train
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe

# --- MetaCentrum training job for the stock YOLO26 veneer-defect model ---
# Submit with:  qsub YOLO26/scripts/train_metacentrum.sh
# Watch with:   qstat -w $USER   /   qstat -f <jobid>

set -euo pipefail
cd "/storage/brno12-cerit/home/drking/diplomka/vevees-defects/YOLO26"

# 1. Load Mambaforge (MetaCentrum's standard tool for modern Python/CUDA stacks).
module purge
module add mambaforge

# 2. Reuse a manually-built persistent venv (torch + ultralytics
#    pip-installed by hand) -- run_all.sh just activates it and fails
#    loudly if incomplete.
export VENV_DIR="/storage/brno12-cerit/home/drking/.conda/envs/yolo26"

# 3. Sanity-check CUDA before doing anything else -- a broken env would
#    otherwise fail late, mid-training.
# shellcheck disable=SC1091
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in venv — rebuild it with a cu121+ torch wheel'"
    deactivate
fi

# 4. Run the pipeline: venv activation, shared dataset, fine-tuning,
#    evaluation on this project's veneer photos.
DEVICE=0 MODEL=yolo26s.pt EPOCHS=100 BATCH=32 IMGSZ=640 ./run_all.sh

echo "JOB DONE: $(ls -t runs/detect/wood_defects_yolo26*/weights/best.pt 2>/dev/null | head -1)"
