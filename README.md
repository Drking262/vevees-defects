# vevees-defects: wood veneer defect detection pipeline

Detects wood veneer defects with three local tools, one per defect group: a
YOLO classifier for localized blemishes, a color-analysis tool for
whole-panel color defects, and a texture/orientation tool for figure-cut
type. `pipeline/analyze.py` runs all three and prints one combined report.

## Three ways to find localized defects

- **Classification** (`pipeline/train.py` + `pipeline/infer.py`): whole-image
  label, no annotation needed, works today. Pair with `pipeline/gradcam.py`
  for an approximate "where" heatmap.
- **Detection** (`pipeline/annotate.py` + `pipeline/train_detect.py` +
  `pipeline/infer_detect.py`): a real bounding box, from a model trained on
  box coordinates -- see "Real detection" below for the annotation step.
- **Zero-shot** (`pipeline/infer_zeroshot.py`): pretrained open-vocabulary
  detector (YOLO-World), no training needed -- doesn't actually work on this
  dataset, see below.

`pipeline/analyze.py` uses detection once trained
(`runs/detect/defects/weights/best.pt` exists), else falls back to
classification.

## Dataset reality

`data/set01/` and `data/set02/` each hold one macro photo (4500x6000) per
defect sample, named after the defect (e.g. `dub_vypadavy_suk_1.jpg` =
falling-out knot). No box annotations -- filename is the only label, treated
as ground truth. So the pipeline fine-tunes a **YOLO classification head**,
not a detector, on whole images.

Only discrete, localized blemishes go into YOLO:

| class | meaning |
|---|---|
| `vypadavy_suk` | falling-out knot |
| `zarostle_suky` | overgrown/ingrown knots |
| `cerny_soucek_drevokazny_hmyz` | black knot + wood-boring insect damage |
| `vylomeni` | breakout / chipped-out chunk |
| `hniloba` | rot |

Excluded (handled by the classical tools below instead): `fladrova_dyha`,
`polofladrova_dyha`, `rovnoleta_dyha`, `podelna_neodlupciva_zrcatka`
(whole-panel grain/figure), `barevny_rozdil_stredni`, `rozbarvenost`
(whole-panel/seam color), `bel` (sapwood, not a discrete blemish), `sval`,
`zabeh` (whole-panel texture). Rationale per class in `pipeline/config.py`
(`INCLUDED_CLASSES`).

**Data is tiny**: the 5 included classes have 6, 3, 1, 2, 3 unique images
(~15 total) -- enough to stand up the pipeline end-to-end, not enough for a
reliable detector. Treat results as a smoke test.

## Setup

No GPU here, so install CPU-only torch first (plain `pip install
ultralytics` pulls the ~5GB CUDA stack):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

Model: **YOLOv8** (`yolov8n-cls.pt`). An initial pass used YOLO26, but its
Grad-CAM heatmaps were unreliable, so the pipeline switched to YOLOv8.
Weights auto-download on first use.

## Usage

```bash
# 1. Build dataset/train, dataset/val from data/set01, data/set02
python -m pipeline.prepare_dataset

# 2. Fine-tune yolov8n-cls on it (CPU)
python -m pipeline.train                 # defaults: 60 epochs, imgsz 640
python -m pipeline.train --epochs 100 --model yolov8s-cls.pt

# 3. Classify new images
python -m pipeline.infer path/to/photo.jpg
python -m pipeline.infer path/to/folder/ --weights runs/classify/defects/weights/best.pt
```

Training runs land in `runs/classify/<name>/`; `dataset/` and `runs/` are
git-ignored (regenerable from `data/`).

### Seeing *where* a prediction came from (Grad-CAM)

The classifier only outputs a whole-image label. `pipeline/gradcam.py`
computes a Grad-CAM heatmap from the last spatial feature map instead:

```bash
python -m pipeline.gradcam path/to/photo.jpg                  # -> gradcam_out/<name>_gradcam.jpg
python -m pipeline.analyze path/to/photo.jpg --gradcam         # same, alongside the full report
```

Approximation, not a validated box: lands correctly on the knot classes, but
missed the visible edge on a `vylomeni` (breakout) test photo -- likely a
data limitation (only 2 training photos for that class), not a model bug.
Drawn on the center-cropped view the model actually sees, so an edge defect
on a tall photo can fall outside the crop.

### Real detection: an actual box, from a model trained on boxes

A photo can show more than one defect (e.g. one black-knot box plus several
separate insect holes), so detection classes are finer-grained than
classification labels -- `cerny_soucek` and `drevokazny_hmyz` are split even
though classification lumps them into one label (see
`pipeline/config.DETECT_CLASS_NAMES`).

```bash
# 1. Draw boxes on each photo, in your browser
python -m pipeline.annotate
# -> open http://localhost:8765, click-drag a box, pick its class, "Save & Next".
#    Resumable -- Ctrl+C and rerun any time.

# 2. Package the boxes + photos into a YOLO detection dataset
python -m pipeline.prepare_detect_dataset

# 3. Fine-tune a real YOLOv8 detector on them (CPU)
python -m pipeline.train_detect            # defaults: 100 epochs, imgsz 640

# 4. Detect + draw boxes on new photos
python -m pipeline.infer_detect path/to/photo.jpg     # -> detect_out/<name>.jpg
python -m pipeline.analyze path/to/photo.jpg          # uses detection automatically once trained
```

`annotations/boxes.json` is manual work product, so unlike `dataset/` and
`runs/` it's **not** git-ignored -- commit it once annotated.

Same small-data caveat as the classifier: with 1-3 boxes per class,
`prepare_detect_dataset.py` falls back to mirroring train into val whenever
a class doesn't clear the holdout threshold, so metrics are a smoke test,
not a generalization estimate.

### Zero-shot with a pretrained detector (`pipeline/infer_zeroshot.py`)

[YOLO-World](https://docs.ultralytics.com/models/yolo-world/) draws a box
around anything named at inference time (e.g. `"wood knot"`), no
fine-tuning needed. Weights auto-download.

```bash
python -m pipeline.infer_zeroshot path/to/photo.jpg
python -m pipeline.infer_zeroshot path/to/folder/ --conf 0.01
python -m pipeline.infer_zeroshot path/to/photo.jpg --classes "knot,crack,hole"
```

**Doesn't work on this dataset.** Across every photo, prompt, and both 640
and 1280 `imgsz`, the highest confidence YOLO-World ever assigns is 0.051 --
noise, far below the 0.25 default threshold. It was pretrained on
natural/web imagery, and close-up oak grain doesn't resemble what it learned
"hole" or "crack" to mean. Kept as a documented negative result, left out of
`pipeline/analyze.py`'s report.

## The other anomalies: color and grain/figure tools

Classes YOLO excludes have even less data (1-2 photos each), so instead of
training a classifier that would just memorize them, these tools compute
the property directly from the image -- no training needed, and they get
more accurate (not "trainable") as reference photos are added.

### Color anomalies (`pipeline/color_anomaly.py`)

Covers `bel`, `rozbarvenost`, `barevny_rozdil_stredni`. Converts to CIELAB,
takes the median color of image patches/halves (median so a knot doesn't
skew the reading), and flags:

- `barevny_rozdil`: large color difference between image quadrants
- `rozbarvenost`: left/right halves don't match (glued-strip seam)
- `bel`: lighter, less saturated patch near the panel's edge

```bash
python -m pipeline.color_anomaly path/to/photo.jpg
python -m pipeline.color_anomaly path/to/folder/
python -m pipeline.color_anomaly --calibrate   # dump metrics for every photo in data/, for threshold-tuning
```

Thresholds were eyeballed from `--calibrate` output, not fit on held-out
data -- expect to revisit as more photos arrive.

### Veneer figure-cut type (`pipeline/grain_pattern.py`)

Covers `fladrova_dyha`, `polofladrova_dyha`, `rovnoleta_dyha`,
`podelna_neodlupciva_zrcatka` -- which grain/figure pattern the veneer was
cut to show, via nearest-neighbor match against reference photos (LBP
texture histogram + structure-tensor orientation, see
`pipeline/grain_features.py`), not a trained classifier.

```bash
python -m pipeline.grain_pattern path/to/photo.jpg
python -m pipeline.grain_pattern --evaluate     # leave-one-out accuracy over the 8 reference photos
python -m pipeline.grain_pattern --calibrate    # dump orientation metrics for every photo in data/
```

**Result**: `--evaluate` gets 3/8 right. `zrcatka` is reliably distinguished
(lower grain-orientation coherence), but `fladrova`/`polofladrova`/`rovnoleta`
form a continuum that 2 photos/class can't reliably separate -- more photos
and a feature that directly measures cathedral-figure curvature would help
more than better thresholds.

The same math also reports `runout_deg` (grain deviation from vertical,
targets `zabeh`) and `orientation_std_deg` (local orientation variance,
flags knot swirls) as raw numbers, not flags -- the two `zabeh` reference
photos disagreed with each other (12.4° vs 1.2°), so no threshold is
justified yet.

### Combined report

```bash
python -m pipeline.analyze path/to/photo.jpg
python -m pipeline.analyze path/to/folder/ --weights runs/classify/defects/weights/best.pt
```

Runs the classifier (if trained), color_anomaly, and grain_pattern, and
prints all three -- no arbitration needed since each owns a disjoint set of
classes.

## Next steps

- Collect more photos per class -- the binding constraint for every tool
  here, not the model or pipeline.
- Run `python -m pipeline.annotate` and train the detector if you haven't --
  the difference between a label and an actual box on output.
- New photos with multiple defects per frame need multiple boxes;
  `pipeline/annotate.py` currently assumes one dominant defect per photo.
- The figure-cut continuum likely needs a purpose-built curvature feature
  rather than more generic texture descriptors.
