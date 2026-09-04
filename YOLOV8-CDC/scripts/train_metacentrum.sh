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

# 2. Reuse the persistent PTCG venv in shared storage instead of building a
#    fresh one -- it already has a working cu-build torch, so run_all.sh
#    only needs to add the YOLO-specific deps (ultralytics etc.) on top.
export VENV_DIR="/storage/brno12-cerit/home/drking/.conda/envs/ptcg"

# 3. Sanity-check the env has a working CUDA build before doing anything
#    else -- a broken/empty env would otherwise fail late, mid-training.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in venv — rebuild it with a cu121 torch wheel'"
deactivate

# 4. Run the pipeline. run_all.sh handles: venv activation (reusing
#    VENV_DIR since it already exists, so no recreation), the ultralytics
#    patch, dataset prep (the ~4k-image subset ships vendored in the repo,
#    so no download needed), training, and evaluation on this project's
#    own veneer photos.
DEVICE=0 SHARDS=1 EPOCHS=150 BATCH=32 IMGSZ=640 ./run_all.sh

echo "JOB DONE: $(ls -t runs/detect/wood_defects_cdc*/weights/best.pt 2>/dev/null | head -1)"
