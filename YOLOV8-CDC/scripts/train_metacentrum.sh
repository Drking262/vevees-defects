#!/bin/bash
#PBS -N vevees_yolo_train
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe

# --- MetaCentrum training job for the YOLOV8-CDC veneer-defect model ---
# Submit with:  qsub YOLOV8-CDC/scripts/train_metacentrum.sh
# Watch with:   qstat -w $USER   /   qstat -f <jobid>

set -euo pipefail
cd "/storage/brno12-cerit/home/drking/diplomka/vevees-defects/YOLOV8-CDC"

# 1. Load Mambaforge (MetaCentrum's standard tool for modern Python/CUDA stacks).
module purge
module add mambaforge

# 2. Reuse the persistent PTCG venv (already has a working CUDA torch) so
#    run_all.sh only adds the YOLO-specific deps on top.
export VENV_DIR="/storage/brno12-cerit/home/drking/.conda/envs/ptcg"

# 3. Sanity-check CUDA before doing anything else -- a broken env would
#    otherwise fail late, mid-training.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in venv — rebuild it with a cu121 torch wheel'"
deactivate

# 4. Run the pipeline: venv reuse, ultralytics patch, dataset prep, train,
#    evaluate on this project's veneer photos.
DEVICE=0 SHARDS=1 EPOCHS=150 BATCH=32 IMGSZ=640 ./run_all.sh

echo "JOB DONE: $(ls -t runs/detect/wood_defects_cdc*/weights/best.pt 2>/dev/null | head -1)"
