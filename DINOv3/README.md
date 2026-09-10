# DINOv3

Qualitative exploration, not a defect detector: runs Meta's [DINOv3](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)
(self-supervised ViT, no fine-tuning) over `../data/set01`/`set02` and
renders its internal patch features -- a PCA of patch tokens down to 3
components, shown as an RGB map next to the original photo. Regions the
model treats as similar (grain, knots, defects) end up the same color. Goal:
see whether an off-the-shelf backbone already separates defects from clean
wood before investing in fine-tuning.

A third panel adds a [PatchCore](https://arxiv.org/abs/2106.08265)-style
anomaly heatmap: a coreset memory bank is built from a pool of patches
(`--coreset-ratio`, default 10%), then every patch is scored by L2 distance
to its nearest bank neighbor. This dataset has no defect-free photos, but it
does have whole-panel/figure-cut classes with no discrete blemish
(`pipeline/config.py`'s `EXCLUDED_CLASSES`). Picked with `--bank-mode`:

- **`reference` (default)**: bank is pooled only from those non-defect
  photos, then defect photos are scored against it. An earlier version
  pooled from *every* photo, but greedy coreset selection preferentially
  picks outliers (i.e. the defects) into their own "normal" bank, so
  defects matched themselves and produced flat noise with no signal.
  Restricting the bank to reference photos fixes that at the root.
- **`per-image`**: bank is built from a single photo's own patches -- same
  contamination problem, per-image instead of dataset-wide. Kept as a
  fallback/ablation (`bank-perimage` vs `bank-reference` output tags).

Works as an anomaly cue because a veneer photo is mostly repetitive grain
with one localized defect, so grain patches find a close bank match while
the defect doesn't. Still a rougher signal than true PatchCore -- a flat
defect (knot-fill) can score *lower* than noisy grain -- treat it as a
qualitative lens, not a threshold-able score. `test_visualize_dino.py`
checks the bank bookkeeping itself (a query photo's patches never enter the
bank; a reference patch that lands in the bank doesn't trivially match
itself at distance 0).

## Setup

DINOv3 weights are **gated** on Hugging Face:

1. Request access (approval isn't instant): https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
2. `pip install transformers` (also pulls `huggingface_hub`'s `hf` CLI)
3. `hf auth login` with a token from https://huggingface.co/settings/tokens

## Usage

```bash
python visualize_dino.py              # all of ../data/set01 + set02 -> dino_out/, reference bank
python visualize_dino.py --limit 2    # smoke test on 2 photos first
python visualize_dino.py path/to/one/image.jpg
python visualize_dino.py --coreset-ratio 0.05   # smaller PatchCore memory bank
python visualize_dino.py --bank-mode per-image  # score each photo only against itself
```

Uses the smallest checkpoint, `dinov3-vits16-pretrain-lvd1689m` (21M
params), by default -- no GPU/limited disk here; pass `--model
facebook/dinov3-vitb16-pretrain-lvd1689m` (or larger) for a sharper map.

Output: one `{set}_{name}_{model}.png` per photo in `--out` (default
`dino_out/`), original + PCA map side by side.

## While DINOv3 access is pending: DINOv2

DINOv2 is Apache-2.0, ungated, and close enough to sanity-check the approach
meanwhile -- point `--model` at `facebook/dinov2-with-registers-small` (see
below for why "with-registers").

## Noisy/speckled PCA map? Use a "with-registers" checkpoint

Plain `dinov2-small`'s PCA looks like random static in a grid pattern.
That's DINOv2's own [high-norm "artifact" tokens](https://arxiv.org/abs/2309.16588):
a few patch tokens carry global image info instead of local content, and
PCA picks them up as noise -- not something fine-tuning would fix.

Fix: `facebook/dinov2-with-registers-small` instead of `dinov2-small` --
dedicated register tokens absorb that global-info role, leaving the patch
grid clean. Confirmed on this dataset: a knot photo went from speckled
background to a smooth gradient with the knot as one clean blob. DINOv3 has
registers built in from the start, so this isn't needed there.
