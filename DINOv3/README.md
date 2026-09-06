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

## Setup

DINOv3 weights are **gated** on Hugging Face -- unlike YOLOv8/YOLO26's
auto-downloading checkpoints, you must request access and authenticate
before the first run:

1. Request access on the model page (approval isn't instant, budget a few
   days): https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
2. Once approved, install the one new dependency (torch/torchvision/numpy/
   scikit-learn/Pillow are already used elsewhere in this repo and installed
   system-wide; see the repo root `README.md`/`requirements.txt`) -- this
   also pulls in `huggingface_hub`, which provides the `hf` CLI used next:

   ```bash
   pip install transformers
   ```
3. Authenticate with a token from https://huggingface.co/settings/tokens:

   ```bash
   hf auth login
   ```

## Usage

```bash
python visualize_dino.py              # all of ../data/set01 + set02 -> dino_out/
python visualize_dino.py --limit 2    # smoke test on 2 photos first
python visualize_dino.py path/to/one/image.jpg
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
