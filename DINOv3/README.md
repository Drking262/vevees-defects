# DINOv3

Qualitative exploration, not a defect detector: runs Meta's [DINOv3](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)
(a self-supervised ViT, no labels/fine-tuning involved) over this project's
oak veneer macro photos in `../data/set01` and `../data/set02`, and renders
what its internal patch features look like -- a PCA of the patch tokens down
to 3 components, shown as an RGB map next to the original photo. Regions the
model treats as similar (grain direction, knots, defects, ...) end up the
same color. The point is to see whether an off-the-shelf, zero-training
vision backbone already separates defects from clean wood, before investing
in fine-tuning anything.

A third panel adds a [PatchCore](https://arxiv.org/abs/2106.08265)-style
anomaly heatmap over the same patch features: a coreset memory bank is built
from a pool of patches (`--coreset-ratio`, default 10% of the pool), then
every patch is scored by its L2 distance to its nearest coreset neighbor.
PatchCore's original memory bank comes from a separate set of known-good
training photos; this dataset has no photos labeled defect-free, but it does
have whole-panel/figure-cut classes with no discrete blemish (`bel`,
`rovnoleta_dyha`, `sval`, ... -- `pipeline/config.py`'s own
`EXCLUDED_CLASSES`, already excluded from the YOLO classifier for the same
"not a localized blemish" reason). Picked with `--bank-mode`:

- **`reference` (default)**: the bank is pooled only from those
  non-localized-defect photos (`pipeline/config.py`'s `INCLUDED_CLASSES`
  decides which photos are "defect" vs "reference" by filename, same lookup
  the classifier pipeline uses); the localized-defect photos are then scored
  against that bank. This matters more than it might sound: an earlier
  version of this pooled patches from *every* photo, defects included --
  greedy coreset selection specifically hunts for outliers, so it
  preferentially picked the defects themselves *into* their own "normal"
  bank, and they then matched themselves. The result looked like uniform
  speckled noise with no response at the actual defect (confirmed on
  `dub_cerny_soucek_+_drevokazny_hmyz.jpg`, whose obvious black knot got no
  distinguishable signal at all). Restricting the bank to reference-only
  photos fixes that at the root: a defect photo never contributes any patch
  to the bank, so there's nothing for its own defect to match. Runs in two
  passes: extract every photo's patch tokens first (one forward pass each,
  same total model cost as before), build one coreset bank from the pooled
  *reference* patches only (`--max-pool`, default 20000, randomly
  subsamples the reference pool first if it's bigger than that -- greedy
  coreset selection doesn't scale to the full pool on CPU), then score +
  plot every photo (reference and defect alike) against that bank.
- **`per-image`** (the original behavior): the bank is built from a single
  photo's own patches, so a defect photo is scored against a bank that
  includes its own defect -- the same contamination problem, just per-image
  instead of dataset-wide. Kept as a fallback/ablation -- output filenames
  are tagged `bank-reference` / `bank-perimage` so the two modes' outputs
  don't overwrite each other.

It still works as an anomaly cue because a veneer photo is mostly
repetitive grain background with one localized defect, so grain patches
find a close match in the reference bank while the defect doesn't. Even
with a clean reference-only bank this is a rougher signal than true
PatchCore and won't isolate every defect type cleanly (a flat, uniform
defect like a knot-fill can score *lower* than noisy grain texture) --
treat it as another qualitative lens, not a score to threshold on.
`test_visualize_dino.py` (`python test_visualize_dino.py`, no
torch/transformers needed) checks the bank bookkeeping itself -- in
particular that a *query* (defect) photo's patches can never end up in the
bank even when one is an extreme outlier (the exact bug above), and that a
*reference* photo's own patch, when it happens to land in the bank, doesn't
trivially match itself at distance 0 instead of its true nearest *other*
neighbor.

## Setup

DINOv3 weights are **gated** on Hugging Face -- unlike YOLOv8/YOLO26's
auto-downloading checkpoints, you must request access and authenticate
before the first run:

1. Request access on the model page (approval isn't instant, budget a few
   days): https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
2. Once approved, install the one new dependency (torch/torchvision/numpy/
   scipy/scikit-learn/Pillow are already used elsewhere in this repo and
   installed system-wide; see the repo root `README.md`/`requirements.txt`) --
   this also pulls in `huggingface_hub`, which provides the `hf` CLI used
   next:

   ```bash
   pip install transformers
   ```
3. Authenticate with a token from https://huggingface.co/settings/tokens:

   ```bash
   hf auth login
   ```

## Usage

```bash
python visualize_dino.py              # all of ../data/set01 + set02 -> dino_out/, reference bank
python visualize_dino.py --limit 2    # smoke test on 2 photos first
python visualize_dino.py path/to/one/image.jpg
python visualize_dino.py --coreset-ratio 0.05   # smaller PatchCore memory bank
python visualize_dino.py --bank-mode per-image  # score each photo only against itself (old, weaker behavior)
```

Uses the smallest checkpoint, `dinov3-vits16-pretrain-lvd1689m` (21M params),
by default -- deliberately, since this machine has no GPU and limited disk;
pass `--model facebook/dinov3-vitb16-pretrain-lvd1689m` (or larger) for a
sharper feature map if disk/CPU time allow.

Output: one `{set}_{name}_{model}.png` per input photo in `--out` (default
`dino_out/`), original + PCA feature map side by side. Prefixed by source
folder since `set01/` and `set02/` reuse filenames for the same defect type.

## While DINOv3 access is pending: DINOv2

DINOv3 access is gated and can take days to approve (see above). DINOv2 is
Apache-2.0, ungated, and architecturally close enough to sanity-check the
approach in the meantime -- just point `--model` at it, e.g.
`facebook/dinov2-with-registers-small` (see below for why "with-registers",
not plain `dinov2-small`).

## Noisy/speckled PCA map? Use a "with-registers" checkpoint

Plain `dinov2-small`'s patch-feature PCA looks like random static in a grid
pattern layered over the real signal. That's not a bug in this script -- it's
DINOv2's own [high-norm "artifact" tokens](https://arxiv.org/abs/2309.16588):
a handful of patch tokens repurpose themselves to carry global image info
instead of local content, and PCA-ing raw patch features picks those up as
noise. It is *not* something fine-tuning would fix.

Fix: use a checkpoint trained with dedicated register tokens, which absorb
that global-info role and leave the patch grid clean --
`facebook/dinov2-with-registers-small` (still zero-shot, no training) instead
of `facebook/dinov2-small`. Confirmed directly on this dataset: the same
knot photo went from a speckled background with a barely-there knot signal
to a smooth gradient with the knot as one solid, sharply-bounded blob.
DINOv3 has registers built in from the start (`num_register_tokens=4` in its
config, already handled by this script), so this shouldn't be needed there.
