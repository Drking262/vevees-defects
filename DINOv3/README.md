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
2. Once approved: `pip install huggingface_hub[cli]` (if not already
   present) then `huggingface-cli login` with a token from
   https://huggingface.co/settings/tokens

Then install the one new dependency (torch/torchvision/numpy/scikit-learn/
Pillow are already used elsewhere in this repo and installed system-wide;
see the repo root `README.md`/`requirements.txt`):

```bash
pip install transformers
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

Output: one `{name}_dinov3.png` per input photo in `dino_out/`, original +
PCA feature map side by side.
