"""Visualize what DINOv3 (Meta's self-supervised ViT) picks up on across this
project's oak veneer macro photos -- no training, just a forward pass through
a frozen pretrained backbone. Patch token features are PCA'd down to 3
components and rendered as an RGB map next to the original photo, so regions
the model treats as similar (grain, knots, defects, ...) cluster into the
same color. A third panel adds a PatchCore-style anomaly heatmap over the
same patch features (see `shared_anomaly_map` / `patchcore_anomaly_map`).

Two PatchCore memory-bank modes (`--bank-mode`):
  reference (default) -- real PatchCore requires the bank to come from
    defect-free photos, never from the photos being scored (otherwise greedy
    coreset selection, which specifically hunts for outliers, preferentially
    pools the defects themselves *into* the "normal" bank, and they then
    match themselves -- exactly the failure this mode avoids). This dataset
    has no photos labeled defect-free, but it does have whole-panel/figure-
    cut classes with no discrete blemish (bel, rovnoleta_dyha, sval, ... --
    pipeline/config.py's own EXCLUDED_CLASSES, excluded from the YOLO
    classifier for the same reason). Those become the reference/bank set;
    the 5 localized-defect classes (pipeline/config.py's INCLUDED_CLASSES)
    are the query set actually being scored against that bank.
  per-image -- the original per-photo bank (score a photo only against its
    own patches, including its own defect -- a known weaker signal, see
    patchcore_anomaly_map's docstring). Kept as a fallback/ablation.

Requires a Hugging Face account with access to the gated DINOv3 weights --
see README.md for the one-time request + login step.

Usage:
    python visualize_dino.py                 # runs on ../data/set01 + set02
    python visualize_dino.py path/to/img.jpg
    python visualize_dino.py --model facebook/dinov3-vitb16-pretrain-lvd1689m
    python visualize_dino.py --limit 2        # smoke test on 2 photos
    python visualize_dino.py --bank-mode per-image   # old per-photo bank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# torch/transformers are imported lazily inside the functions that need them
# (extract_patch_tokens, main) so the pure-numpy PatchCore/PCA logic above --
# the part test_visualize_dino.py exercises -- can be imported and tested
# without the (large, gated-model) ML stack installed.

REPO_ROOT = Path(__file__).resolve().parent
VEVEES_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(VEVEES_ROOT))
from pipeline.config import INCLUDED_CLASSES  # noqa: E402 -- the 5 localized-defect classes
from pipeline.dataset_scan import stem_to_class  # noqa: E402

DEFAULT_SOURCES = [VEVEES_ROOT / "data" / "set01", VEVEES_ROOT / "data" / "set02"]
DEFAULT_OUT_DIR = REPO_ROOT / "dino_out"
DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_localized_defect(f: str) -> bool:
    """True if f's filename matches one of pipeline/config.py's 5 localized-
    defect classes (a real blemish to find) rather than a whole-panel/figure-
    cut characteristic (no discrete blemish -- usable as a PatchCore
    reference/"normal" photo instead)."""
    return stem_to_class(Path(f).stem, INCLUDED_CLASSES) is not None


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


def greedy_coreset(features: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
    """Greedy k-center subsampling (the PatchCore paper's coreset method):
    repeatedly pick the point farthest from everything picked so far, so the
    result covers the feature space instead of just resampling the densest
    cluster. Returns indices into `features`, not the vectors themselves --
    the caller needs the indices to exclude trivial self-matches."""
    n = features.shape[0]
    m = max(1, round(n * ratio))
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    min_dists = np.linalg.norm(features - features[selected[0]], axis=1)
    for _ in range(m - 1):
        nxt = int(np.argmax(min_dists))
        selected.append(nxt)
        min_dists = np.minimum(min_dists, np.linalg.norm(features - features[nxt], axis=1))
    return np.array(selected)


def patchcore_anomaly_map(
    patch_tokens: np.ndarray, grid_h: int, grid_w: int, coreset_ratio: float
) -> np.ndarray:
    """PatchCore-lite anomaly score per patch: build a coreset memory bank
    from this same image's own patch features, then score each patch by its
    L2 distance to its nearest coreset neighbor.

    The original PatchCore memory bank comes from a separate set of known-
    defect-free training images; this project has no such labeled "normal"
    set (every photo here is a specific named defect), so the bank is built
    per-image instead. That still works as an anomaly cue because a veneer
    photo is mostly repetitive grain background with one localized defect: a
    grain patch almost always has a near-duplicate elsewhere in the same
    photo (low distance to the bank), while a knot/rot/insect blemish is
    locally unique and stands out as a high-distance blob.
    """
    bank_idx = greedy_coreset(patch_tokens, coreset_ratio)
    dists = cdist(patch_tokens, patch_tokens[bank_idx])
    # A patch chosen for the bank trivially matches itself at distance 0;
    # without excluding that, every bank patch renders as a spurious "zero
    # anomaly" dot scattered across the whole image instead of just the
    # defect standing out.
    dists[bank_idx, np.arange(len(bank_idx))] = np.inf
    nn_dists = dists.min(axis=1)
    lo, hi = nn_dists.min(), nn_dists.max()
    scores = (nn_dists - lo) / max(hi - lo, 1e-6)
    return scores.reshape(grid_h, grid_w)


def build_shared_bank(
    all_patch_tokens: list[np.ndarray],
    eligible: np.ndarray,
    coreset_ratio: float,
    max_pool: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool patches from the eligible ("reference", defect-free) photos only
    and greedy-coreset a shared PatchCore memory bank out of that pool, so
    every photo (reference or query/defect) is scored against what's normal
    across the reference set -- never against defect patches, including its
    own, since a non-eligible (query) photo never contributes to the pool at
    all. `eligible` is a bool array, one entry per photo in all_patch_tokens.

    Returns (bank_features, bank_global_idx, offsets):
      - bank_global_idx indexes into the concatenation of all_patch_tokens
        (i.e. "global patch id"), needed so shared_anomaly_map can exclude a
        reference photo's own patches from trivially matching themselves in
        the bank (a query photo's patches can never appear in the bank, so
        this exclusion is a no-op for query photos).
      - offsets[i] is the global-id of the first patch of photo i (for every
        photo, reference or query), so a photo's local patch index can be
        converted to/from a global id.

    Greedy k-center over the full reference pool (tens of thousands of
    patches) is too slow to run directly -- max_pool bounds it by randomly
    subsampling the pool first when it's larger than that, then
    coreset-selecting the bank from the (capped) pool.
    """
    lengths = [t.shape[0] for t in all_patch_tokens]
    offsets = np.cumsum([0] + lengths)[:-1]
    pooled = np.concatenate(all_patch_tokens, axis=0)

    eligible_global_idx = np.nonzero(np.repeat(np.asarray(eligible), lengths))[0]
    if eligible_global_idx.size == 0:
        raise SystemExit(
            "no reference (non-localized-defect) photos among the given files -- "
            "shared PatchCore bank needs at least one; pass a broader source, or "
            "use --bank-mode per-image instead"
        )

    rng = np.random.default_rng(seed)
    pool_idx = (
        rng.choice(eligible_global_idx, size=max_pool, replace=False)
        if eligible_global_idx.size > max_pool
        else eligible_global_idx
    )
    bank_pool_idx = greedy_coreset(pooled[pool_idx], coreset_ratio, seed=seed)
    bank_global_idx = pool_idx[bank_pool_idx]
    return pooled[bank_global_idx], bank_global_idx, offsets


def shared_anomaly_map(
    patch_tokens: np.ndarray,
    image_offset: int,
    bank_features: np.ndarray,
    bank_global_idx: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Same L2-nearest-neighbor scoring as patchcore_anomaly_map, but against
    the shared cross-image bank -- excluding any bank entry that happens to
    be one of this same photo's own patches (a trivial, self-inflated
    zero-distance match), same fix as patchcore_anomaly_map's bank_idx
    exclusion but generalized across images.
    """
    dists = cdist(patch_tokens, bank_features)
    local = bank_global_idx - image_offset
    own = (local >= 0) & (local < patch_tokens.shape[0])
    if own.any():
        cols = np.nonzero(own)[0]
        dists[local[own], cols] = np.inf
    nn_dists = dists.min(axis=1)
    lo, hi = nn_dists.min(), nn_dists.max()
    scores = (nn_dists - lo) / max(hi - lo, 1e-6)
    return scores.reshape(grid_h, grid_w)


def extract_patch_tokens(f: str, processor, model, model_name: str, long_side: int, patch_size: int, num_registers: int):
    """One forward pass through the frozen backbone for one photo. Returns
    (resized_image, patch_tokens, grid_h, grid_w)."""
    import torch

    image = load_resized(f, long_side, patch_size)
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
            f"wrong for {model_name}, check model.config"
        )
    return image, patch_tokens, grid_h, grid_w


def save_panel(f: str, image, rgb_map: np.ndarray, anomaly_map: np.ndarray, grid_w: int, grid_h: int,
               model_name: str, anomaly_title: str, out_dir: Path, tag: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title(Path(f).name, fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(rgb_map)
    axes[1].set_title(f"{model_name} patch-feature PCA ({grid_w}x{grid_h} patches)", fontsize=9)
    axes[1].axis("off")
    w, h = image.size
    axes[2].imshow(image)
    axes[2].imshow(anomaly_map, cmap="jet", alpha=0.45, extent=(0, w, h, 0), interpolation="bilinear")
    axes[2].set_title(anomaly_title, fontsize=9)
    axes[2].axis("off")
    fig.tight_layout()

    # set01/ and set02/ reuse filenames for the same defect type (e.g. both
    # have a dub_bel.jpg) -- prefix with the parent dir so outputs from the
    # two sets don't silently overwrite each other. tag (bank mode) is in the
    # filename too, so reference- and per-image-bank runs don't overwrite
    # each other either.
    model_slug = model_name.split("/")[-1]
    out_path = out_dir / f"{Path(f).parent.name}_{Path(f).stem}_{model_slug}_{tag}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="*", default=[str(s) for s in DEFAULT_SOURCES])
    p.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--long-side", type=int, default=768, help="resize longest side to ~this many px before feeding the model")
    p.add_argument("--limit", type=int, default=None, help="only process the first N images (smoke test)")
    p.add_argument(
        "--bank-mode",
        choices=["reference", "per-image"],
        default="reference",
        help="PatchCore memory bank: 'reference' (default) builds the bank only from "
             "whole-panel/figure-cut photos with no discrete blemish, then scores the "
             "localized-defect photos against it; 'per-image' scores a photo only "
             "against its own patches (weaker, includes the defect in its own bank)",
    )
    p.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.1,
        help="fraction of the bank's source pool kept as the PatchCore memory bank "
             "(the pooled reference-photo patches in reference mode, one photo's patches in per-image mode)",
    )
    p.add_argument(
        "--max-pool",
        type=int,
        default=20000,
        help="reference mode only: cap on how many pooled reference patches greedy coreset selection runs over "
             "(randomly subsampled down to this first if there are more) -- keeps bank-building tractable on CPU",
    )
    args = p.parse_args()

    files = resolve_images([Path(s) for s in args.source])
    if not files:
        raise SystemExit("no images found")
    if args.limit:
        files = files[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoImageProcessor, AutoModel

    print(f"loading {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    patch_size = model.config.patch_size
    num_registers = getattr(model.config, "num_register_tokens", 0)

    if args.bank_mode == "per-image":
        for f in files:
            image, patch_tokens, grid_h, grid_w = extract_patch_tokens(
                f, processor, model, args.model, args.long_side, patch_size, num_registers
            )
            rgb_map = patch_pca_rgb(patch_tokens, grid_h, grid_w)
            anomaly_map = patchcore_anomaly_map(patch_tokens, grid_h, grid_w, args.coreset_ratio)
            out_path = save_panel(
                f, image, rgb_map, anomaly_map, grid_w, grid_h, args.model,
                f"PatchCore anomaly, per-image bank (coreset ratio={args.coreset_ratio})",
                out_dir, tag="bank-perimage",
            )
            print(f"wrote {out_path}")
        return

    # reference mode: pass 1 extracts every photo's patch tokens (one forward
    # pass per photo, same total model compute as per-image mode -- just done
    # up front instead of interleaved with plotting), then one bank is built
    # from only the reference (non-localized-defect) photos among them, then
    # pass 2 scores + plots every photo against that bank.
    extracted = []
    for i, f in enumerate(files, 1):
        image, patch_tokens, grid_h, grid_w = extract_patch_tokens(
            f, processor, model, args.model, args.long_side, patch_size, num_registers
        )
        extracted.append((f, image, patch_tokens, grid_h, grid_w))
        print(f"extracted features {i}/{len(files)}: {f}")

    eligible = np.array([not is_localized_defect(e[0]) for e in extracted])
    n_ref, n_query = int(eligible.sum()), int((~eligible).sum())
    print(f"{n_ref} reference photo(s) (bank source), {n_query} localized-defect photo(s) (scored) ...")

    print(f"building reference PatchCore bank (coreset_ratio={args.coreset_ratio}, max_pool={args.max_pool}) ...")
    bank_features, bank_global_idx, offsets = build_shared_bank(
        [e[2] for e in extracted], eligible, args.coreset_ratio, args.max_pool
    )
    print(f"bank size: {len(bank_global_idx)} patches")

    for (f, image, patch_tokens, grid_h, grid_w), offset in zip(extracted, offsets):
        rgb_map = patch_pca_rgb(patch_tokens, grid_h, grid_w)
        anomaly_map = shared_anomaly_map(patch_tokens, offset, bank_features, bank_global_idx, grid_h, grid_w)
        out_path = save_panel(
            f, image, rgb_map, anomaly_map, grid_w, grid_h, args.model,
            f"PatchCore anomaly, reference bank ({len(bank_global_idx)} patches from {n_ref} reference photos)",
            out_dir, tag="bank-reference",
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
