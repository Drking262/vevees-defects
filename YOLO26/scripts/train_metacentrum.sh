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

# 2. Reuse a persistent venv in shared storage instead of building a fresh
#    one every job -- but a DEDICATED one, not the ptcg env the YOLOV8-CDC
#    job reuses: that env is pinned to ultralytics==8.1.0 for the CDC
#    architecture patch, while YOLO26 needs a current ultralytics release.
#    Sharing one env between the two jobs would mean whichever runs last
#    silently changes the other's ultralytics version. First run here builds
#    it (needs network); every run after reuses it (no network needed).
export VENV_DIR="/storage/brno12-cerit/home/drking/.conda/envs/yolo26"

# 3. Sanity-check the env has a working CUDA build before doing anything
#    else -- a broken/empty env would otherwise fail late, mid-training.
#    On the very first run (env doesn't exist yet) this legitimately fails --
#    run run_all.sh once by hand first (or just resubmit; run_all.sh creates
#    and populates VENV_DIR, so a 2nd submission passes this check).
# shellcheck disable=SC1091
if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
    python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in venv — rebuild it with a cu121+ torch wheel'"
    deactivate
fi

# 4. Run the pipeline. run_all.sh handles: venv creation/activation, deps,
#    the shared dataset (vendored in ../YOLOV8-CDC, no download needed),
#    fine-tuning from the vendored yolo26s.pt checkpoint (no download
#    needed), and evaluation on this project's own veneer photos.
DEVICE=0 MODEL=yolo26s.pt EPOCHS=100 BATCH=32 IMGSZ=640 ./run_all.sh

echo "JOB DONE: $(ls -t runs/detect/wood_defects_yolo26*/weights/best.pt 2>/dev/null | head -1)"
