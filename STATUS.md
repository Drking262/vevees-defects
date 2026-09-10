# Current status

What's working vs. still in progress across this repo's pieces, at a glance.
Each subproject's own README has the full detail; this is just the map.

| component | status | notes |
|---|---|---|
| `pipeline/` -- classification, detection, zero-shot, color/grain tools | Working | End-to-end on `data/set01`+`set02`; see top-level README for per-tool caveats (tiny dataset, zero-shot doesn't transfer well). |
| `annotations/` -- box annotation tool + `boxes.json` | Working | 10 boxes annotated so far via `pipeline/annotate.py`. |
| `YOLOV8-CDC/` -- custom architecture fine-tuned on the public wood-defects dataset | Working | CPU smoke test verified end-to-end (build/train/eval/checkpoint); sample detections in `smrk/`. Real training run needs a GPU (MetaCentrum). |
| `DINOv3/` -- patch-feature visualization (PCA + PatchCore anomaly map) | Working | Runs end-to-end (DINOv2 fallback, since DINOv3 needs gated HF access); PatchCore anomaly-map panel produced noisy/low-signal output so its images were dropped, but the tool itself runs. |
| `YOLO26/` -- stock Ultralytics YOLO26 fine-tuned on the same public dataset | **WIP** | Blocked: MetaCentrum GPU-node venv can't import torch at train time (`scripts/check_env.sh` is the current diagnostic step, not yet resolved). CPU path untested pending this fix. |

## Next steps
See each subproject's README ("Next steps" / "Honest expectations" sections)
for the per-tool detail; the one active blocker across all of them is
YOLO26's MetaCentrum torch import failure above.
