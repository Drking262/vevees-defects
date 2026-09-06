"""Visualize what DINOv3 (Meta's self-supervised ViT) picks up on across this
project's oak veneer macro photos -- no training, just a forward pass through
a frozen pretrained backbone. Patch token features are PCA'd down to 3
components and rendered as an RGB map next to the original photo, so regions
the model treats as similar (grain, knots, defects, ...) cluster into the
same color.

Requires a Hugging Face account with access to the gated DINOv3 weights --
see README.md for the one-time request + login step.

Usage:
    python visualize_dino.py                 # runs on ../data/set01 + set02
    python visualize_dino.py path/to/img.jpg
    python visualize_dino.py --model facebook/dinov3-vitb16-pretrain-lvd1689m
    python visualize_dino.py --limit 2        # smoke test on 2 photos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path(__file__).resolve().parent
VEVEES_ROOT = REPO_ROOT.parent
DEFAULT_SOURCES = [VEVEES_ROOT / "data" / "set01", VEVEES_ROOT / "data" / "set02"]
DEFAULT_OUT_DIR = REPO_ROOT / "dino_out"
DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_images(paths: list[Path]) -> list[str]:
    files: list[str] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(str(f) for f in p.glob("*") if f.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            files.append(str(p))
    return files


def load_resized(path: str, long_side: int, patch_size: int) -> Image.Image:
    """Resize so both dimensions are multiples of patch_size (the model can only
    tile the image into whole patches), preserving aspect ratio via the long side."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = long_side / max(w, h)
        new_w = max(patch_size, round(w * scale / patch_size) * patch_size)
        new_h = max(patch_size, round(h * scale / patch_size) * patch_size)
        return im.resize((new_w, new_h), Image.LANCZOS)


def patch_pca_rgb(patch_tokens: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """PCA the (N, C) patch features to 3 components and min-max each to [0,1]
    so they can be shown as an RGB image of shape (grid_h, grid_w, 3)."""
    comps = PCA(n_components=3).fit_transform(patch_tokens)
    lo, hi = comps.min(axis=0, keepdims=True), comps.max(axis=0, keepdims=True)
    comps = (comps - lo) / np.clip(hi - lo, 1e-6, None)
    return comps.reshape(grid_h, grid_w, 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="*", default=[str(s) for s in DEFAULT_SOURCES])
    p.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--long-side", type=int, default=768, help="resize longest side to ~this many px before feeding the model")
    p.add_argument("--limit", type=int, default=None, help="only process the first N images (smoke test)")
    args = p.parse_args()

    files = resolve_images([Path(s) for s in args.source])
    if not files:
        raise SystemExit("no images found")
    if args.limit:
        files = files[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    patch_size = model.config.patch_size
    num_registers = getattr(model.config, "num_register_tokens", 0)

    for f in files:
        image = load_resized(f, args.long_side, patch_size)
        grid_w, grid_h = image.size[0] // patch_size, image.size[1] // patch_size

        inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
        with torch.inference_mode():
            outputs = model(**inputs)
        tokens = outputs.last_hidden_state[0]  # (1 cls + num_registers + grid_h*grid_w, C)
        patch_tokens = tokens[1 + num_registers :].numpy()
        expected = grid_h * grid_w
        if patch_tokens.shape[0] != expected:
            raise SystemExit(
                f"{f}: got {patch_tokens.shape[0]} patch tokens, expected {expected} "
                f"({grid_w}x{grid_h}) -- patch_size/num_register_tokens assumption is "
                f"wrong for {args.model}, check model.config"
            )

        rgb_map = patch_pca_rgb(patch_tokens, grid_h, grid_w)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title(Path(f).name, fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(rgb_map)
        axes[1].set_title(f"{args.model} patch-feature PCA ({grid_w}x{grid_h} patches)", fontsize=9)
        axes[1].axis("off")
        fig.tight_layout()

        # set01/ and set02/ reuse filenames for the same defect type (e.g. both
        # have a dub_bel.jpg) -- prefix with the parent dir so outputs from the
        # two sets don't silently overwrite each other.
        model_slug = args.model.split("/")[-1]
        out_path = out_dir / f"{Path(f).parent.name}_{Path(f).stem}_{model_slug}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
