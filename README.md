# vevees-defects: wood veneer defect detection pipeline

Detects wood veneer defects using three different local tools, one per group
of defects, matching how differently each group actually shows up in a
photo: a YOLO classifier for localized blemishes, a classical color-analysis
tool for whole-panel color defects, and a classical texture/orientation tool
for veneer figure-cut type. `pipeline/analyze.py` runs all three on a photo
and prints one combined report.

## Three ways to find localized defects: classification, real detection, or zero-shot

There are now three options for the 5 localized-defect classes below:

- **Classification** (`pipeline/train.py` + `pipeline/infer.py`): whole-image
  label only, no annotation needed, works today. Optionally paired with
  `pipeline/gradcam.py` for an approximate "where" heatmap.
- **Detection** (`pipeline/annotate.py` + `pipeline/train_detect.py` +
  `pipeline/infer_detect.py`): a real bounding box around the defect, from a
  model actually trained on box coordinates -- see "Real detection" below
  for the one-time annotation step this needs.
- **Zero-shot** (`pipeline/infer_zeroshot.py`): a pretrained open-vocabulary
  detector (YOLO-World) prompted with plain-English defect phrases, no
  training or annotation at all -- see "Zero-shot with a pretrained
  detector" below for why this one doesn't actually work on this dataset.

`pipeline/analyze.py` uses detection automatically once it's been trained
(`runs/detect/defects/weights/best.pt` exists), and falls back to
classification otherwise.

## Dataset reality and why classification was the first pass, not detection

`data/set01/` and `data/set02/` each contain one high-resolution macro photo
(4500x6000) per defect sample, named after the defect it shows, e.g.
`dub_vypadavy_suk_1.jpg` = falling-out knot. There are no bounding-box
annotations -- the filename is the only label. Per project decision, that's
treated as the ground truth: **this pipeline fine-tunes a YOLO
classification head on whole images**, not a YOLO detector.

Only defects that are a discrete, spatially localized blemish are included,
since those are what a photo-classifier can plausibly key on:

| class | meaning |
|---|---|
| `vypadavy_suk` | falling-out knot |
| `zarostle_suky` | overgrown/ingrown knots |
| `cerny_soucek_drevokazny_hmyz` | black knot + wood-boring insect damage |
| `vylomeni` | breakout / chipped-out chunk |
| `hniloba` | rot |

Excluded from YOLO (see below for how they're handled instead):
`fladrova_dyha`, `polofladrova_dyha`, `rovnoleta_dyha`,
`podelna_neodlupciva_zrcatka` (whole-panel grain/figure patterns),
`barevny_rozdil_stredni`, `rozbarvenost` (whole-panel/seam color
characteristics), `bel` (sapwood is a broad zone, not a discrete blemish),
`sval`, `zabeh` (whole-panel texture, no visible localized mark in the
sample photos). See `pipeline/config.py` for the exact rationale per class --
edit `INCLUDED_CLASSES` there if this scope should change.

**Data is extremely small**: after removing an exact duplicate file, the 5
included classes have 6, 3, 1, 2, and 3 unique images respectively (~15
images total). This is enough to stand up the full pipeline end-to-end, not
enough to produce a reliable detector -- treat trained results as a
scaffold/smoke-test until more photos per class are collected.

## Setup

This machine has no NVIDIA GPU. `pip install ultralytics` on its own will
happily pull a torch build with the full CUDA stack (~5 GB of `nvidia-*`
packages) that's dead weight here and can fill a small disk. Install the
CPU-only wheels first:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

The model used is **YOLOv8** (`yolov8n-cls.pt`). An initial pass used YOLO26
(Ultralytics' newest generation), but its Grad-CAM heatmaps (see below) were
unreliable, so the pipeline was switched to the more established YOLOv8.
Weights auto-download from Ultralytics on first use.

## Usage

```bash
# 1. Build dataset/train, dataset/val from data/set01, data/set02
python -m pipeline.prepare_dataset

# 2. Fine-tune yolov8n-cls on it (CPU)
python -m pipeline.train                 # defaults: 60 epochs, imgsz 640
python -m pipeline.train --epochs 100 --model yolov8s-cls.pt   # bigger model/longer run

# 3. Classify new images
python -m pipeline.infer path/to/photo.jpg
python -m pipeline.infer path/to/folder/ --weights runs/classify/defects/weights/best.pt
```

Training runs land in `runs/classify/<name>/` (metrics, confusion matrix,
`weights/best.pt`); `dataset/` and `runs/` are git-ignored since both are
regenerable from `data/`.

### Seeing *where* a prediction came from (Grad-CAM)

The classifier above only outputs a whole-image label -- no box, no
highlighted region. To visualize which part of the photo actually drove a
prediction, `pipeline/gradcam.py` computes a Grad-CAM heatmap from the last
spatial feature map before the classification head's pooling layer:

```bash
python -m pipeline.gradcam path/to/photo.jpg                  # -> gradcam_out/<name>_gradcam.jpg
python -m pipeline.analyze path/to/photo.jpg --gradcam         # same, alongside the full report
```

This is an approximation of "where," not a validated box: on the two knot
classes it lands right on (YOLOv8) or around the rim of (YOLO26) the knot,
but on a `vylomeni` (breakout) test photo it fixated on a dark grain streak
instead of the visible torn edge on *both* backbones -- so that particular
miss looks like a data/feature limitation (only 2 `vylomeni` training photos)
rather than a model-choice bug. Worth keeping in mind before treating the
heatmap as ground truth. The heatmap is drawn on the square, center-cropped
view the model actually sees (Resize+CenterCrop, same as training), so a
defect near the top/bottom edge of a tall photo can fall outside the cropped
view shown.

### Real detection: an actual box, from a model trained on boxes

Grad-CAM approximates "where" from a classifier that never saw box
coordinates. For a real box, drawn by a model actually trained to predict
one, three steps:

A photo can show more than one defect (e.g. `dub_cerny_soucek_+_drevokazny_hmyz.jpg`
has one black-knot box plus several separate wood-boring-insect holes
elsewhere in frame), so the detection class list is finer-grained than the
classifier's whole-image buckets -- `cerny_soucek` (black knot) and
`drevokazny_hmyz` (insect damage) are split into two detection classes even
though the classifier lumps them into one `cerny_soucek_drevokazny_hmyz`
label (see `pipeline/config.DETECT_CLASS_NAMES`). The annotator draws as
many boxes as a photo actually needs, each with its own class:

```bash
# 1. Draw boxes on each photo, in your browser
python -m pipeline.annotate
# -> open http://localhost:8765, click-drag a box around each defect the
#    photo shows, pick its class from the dropdown, then "Save & Next".
#    Boxes are saved to annotations/boxes.json as you go (resumable --
#    Ctrl+C and rerun any time, existing boxes reload with the image).

# 2. Package the boxes + photos into a YOLO detection dataset
python -m pipeline.prepare_detect_dataset

# 3. Fine-tune a real YOLOv8 detector on them (CPU)
python -m pipeline.train_detect            # defaults: 100 epochs, imgsz 640

# 4. Detect + draw boxes on new photos
python -m pipeline.infer_detect path/to/photo.jpg     # -> detect_out/<name>.jpg
python -m pipeline.analyze path/to/photo.jpg          # uses detection automatically once trained
```

`annotations/boxes.json` is real manual work product (not regenerable from
`data/`), so unlike `dataset/`, `detect_dataset/`, and `runs/` it is **not**
git-ignored -- worth committing once you've annotated. With only ~10-16
photos this is a few minutes of clicking, not a data-labeling project.

Same caveat as the classifier applies to accuracy: with 1-3 boxes per class,
`prepare_detect_dataset.py` falls back to mirroring train into val (same
logic as `prepare_dataset.py` and for the same reason) whenever some class
doesn't clear the fair-holdout threshold, so early detection-model metrics
are a pipeline smoke test, not a generalization estimate, same as the
classifier's.

### Zero-shot with a pretrained detector (`pipeline/infer_zeroshot.py`)

Both tools above fine-tune on this project's own ~15 photos. A different
option that needs zero wood-specific training data is a pretrained
open-vocabulary detector: [YOLO-World](https://docs.ultralytics.com/models/yolo-world/)
ships weights already trained (on large web image-text datasets, via
Ultralytics) to draw a box around *anything* you name at inference time, e.g.
`model.set_classes(["wood knot", "crack in wood"])`, with no fine-tuning
step. Weights auto-download on first use, same as the classifier/detector
above.

```bash
python -m pipeline.infer_zeroshot path/to/photo.jpg
python -m pipeline.infer_zeroshot path/to/folder/ --conf 0.01
python -m pipeline.infer_zeroshot path/to/photo.jpg --classes "knot,crack,hole"
```

**Honest result: it doesn't work on this dataset.** Tried across every
included defect photo, at both 640 and 1280 `imgsz`, and with both specific
prompts (`"insect hole in wood"`, `"chipped out wood"`) and generic ones
(`"hole"`, `"dark spot"`, `"circle"`), the highest raw confidence YOLO-World
ever assigns to *any* box on *any* photo is 0.051 -- indistinguishable from
noise, nowhere near the 0.25 default threshold a genuine detection would
clear, and not concentrated on the actual defect region even when a stray
box does fire. YOLO-World was pretrained on natural/web imagery (people,
furniture, everyday objects); close-up oak grain texture with a knot or rot
patch doesn't resemble what it learned "hole" or "crack" to mean, so there's
no real signal here to prompt for. Kept in the pipeline as a documented
negative result and left out of `pipeline/analyze.py`'s combined report
since it has nothing reliable to contribute -- worth revisiting if a
larger/newer open-vocabulary checkpoint (or one actually pretrained on
industrial/material surface imagery) becomes available.

## The other anomalies: color and grain/figure tools

The classes YOLO excludes (whole-panel color or texture characteristics)
have even less data -- 1-2 unique reference photos per class -- so instead
of training a second classifier that would just memorize them, both tools
below compute the property directly from the image. Neither needs training
data, so both work immediately and only get *more accurate* (not
*trainable*) as more reference photos are added.

### Color anomalies (`pipeline/color_anomaly.py`)

Covers `bel` (sapwood), `rozbarvenost` (discoloration/seam mismatch),
`barevny_rozdil_stredni` (color difference). Converts to CIELAB, takes the
median color of image patches/halves (median, not mean, so a small knot
inside a patch doesn't skew its color reading), and flags:

- `barevny_rozdil`: large color difference between image quadrants
- `rozbarvenost`: the left half and right half don't match in color (glued-strip seam)
- `bel`: a lighter, less saturated patch region concentrated near the panel's edge

```bash
python -m pipeline.color_anomaly path/to/photo.jpg
python -m pipeline.color_anomaly path/to/folder/
python -m pipeline.color_anomaly --calibrate   # dump metrics for every photo in data/, for threshold-tuning
```

Thresholds were picked by eyeballing `--calibrate` output across the whole
dataset, not fit on held-out data -- expect to revisit them as more photos
come in (`bel` in particular: sapwood only reads as "lighter than the rest
of the panel", which needs a contrasting heartwood tone actually present in
the shot to detect).

### Veneer figure-cut type (`pipeline/grain_pattern.py`)

Covers `fladrova_dyha`, `polofladrova_dyha`, `rovnoleta_dyha`,
`podelna_neodlupciva_zrcatka` -- these describe *which* grain/figure pattern
the veneer was cut to show, learned by nearest-neighbor match against the
reference photos (Local Binary Pattern texture histogram + structure-tensor
grain orientation/coherence -- see `pipeline/grain_features.py`), rather
than a trained classifier.

```bash
python -m pipeline.grain_pattern path/to/photo.jpg
python -m pipeline.grain_pattern --evaluate     # leave-one-out accuracy over the 8 reference photos
python -m pipeline.grain_pattern --calibrate    # dump orientation metrics for every photo in data/
```

**Honest result**: `--evaluate` currently gets 3/8 right. `zrcatka` (mirror
figure) is reliably distinguished -- it has visibly lower grain-orientation
coherence than the others -- but `fladrova`/`polofladrova`/`rovnoleta`
(flame / half-flame / straight-grain) form a continuum that 2 photos/class
and this feature set can't reliably separate. More reference photos and
probably a feature that directly measures "cathedral-figure curvature" would
help more than better thresholds would.

The same orientation math also reports `runout_deg` (deviation of the
dominant grain direction from vertical -- targets `zabeh`) and
`orientation_std_deg` (how much local grain direction varies across the
panel -- flags up sharply on knot swirls). These are printed as raw numbers
rather than turned into a flag: the two `zabeh` reference photos disagreed
with each other (12.4° vs 1.2° runout), so there isn't yet a justified
threshold.

### Combined report

```bash
python -m pipeline.analyze path/to/photo.jpg
python -m pipeline.analyze path/to/folder/ --weights runs/classify/defects/weights/best.pt
```

Runs the YOLO classifier (if trained weights exist), color_anomaly, and
grain_pattern on each image and prints all three reports together. It
doesn't try to arbitrate between them -- each tool owns a disjoint set of
defect classes, so there's nothing to reconcile.

## Next steps

- Collect more photos per defect class -- single digits per class is the
  binding constraint on quality right now, not the model or pipeline, for
  every one of the tools above (classification, detection, and both
  classical tools alike).
- Run `python -m pipeline.annotate` and train the detector (see "Real
  detection" above) if you haven't yet -- it's the difference between a
  label and an actual box on the output.
- New photos with multiple defects per frame would need multiple boxes per
  image; `pipeline/annotate.py` currently assumes one dominant defect per
  photo, matching the current data.
- The figure-cut continuum (`fladrova`/`polofladrova`/`rovnoleta`) likely
  needs a purpose-built feature (e.g. detecting cathedral-arch curvature)
  rather than more generic texture descriptors -- worth a dedicated pass if
  distinguishing these three matters for the thesis.
