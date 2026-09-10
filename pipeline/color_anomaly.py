"""Classical (no-training-data) color-anomaly detector.

Covers whole-panel color classes YOLO excludes: `bel` (sapwood),
`rozbarvenost` (mismatched glued-strip seam), `barevny_rozdil_stredni`
(color difference). Too few reference photos to train on, and these are
directly measurable CIELAB properties, so compute them instead of learning
them.

Method: split image into a patch grid, take each patch's mean CIELAB color:

  - color_diff_max_de: largest pairwise CIEDE2000 distance between patches
    -- high means non-uniform panel.
  - seam_de: CIEDE2000 distance between left/right half means -- targets
    rozbarvenost's mismatched-strip look.
  - sapwood_fraction: fraction of patches lighter + less saturated than
    the panel median (sapwood is pale/dull vs. heartwood).
  - sapwood_edge_fraction: of those, how many sit in the outer
    quarter-columns -- sapwood runs along the trunk's edge.

Thresholds were eyeballed via `--calibrate`, not fit on held-out data.

Usage:
    python -m pipeline.color_anomaly path/to/image_or_folder
    python -m pipeline.color_anomaly --calibrate   # dump metrics for all of data/
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

from pipeline.imageutil import load_rgb

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

GRID = (6, 8)  # fine grid, rows x cols -- used only for the sapwood area estimate
COARSE_GRID = (2, 2)  # quadrants -- used for whole-panel color-difference
MAX_SIDE = 1024

# Calibrated thresholds -- see module docstring.
COLOR_DIFF_DE_THRESHOLD = 9.0   # flag "barevny_rozdil" above this
SEAM_DE_THRESHOLD = 4.0         # flag "rozbarvenost" (seam mismatch) above this
SAPWOOD_FRACTION_THRESHOLD = 0.15
SAPWOOD_EDGE_CONCENTRATION_THRESHOLD = 0.6


@dataclass
class ColorReport:
    path: str
    color_diff_max_de: float
    seam_de: float
    sapwood_fraction: float
    sapwood_edge_fraction: float
    flags: list[str]


def patch_medians(lab: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Per-patch median (not mean) Lab -- robust to a knot/hole covering part
    of a patch, which would otherwise drag a mean far off the surrounding
    wood tone and masquerade as a whole-panel color difference."""
    h, w, _ = lab.shape
    rows, cols = grid
    row_splits = np.array_split(np.arange(h), rows)
    col_splits = np.array_split(np.arange(w), cols)
    meds = np.zeros((rows, cols, 3))
    for i, rs in enumerate(row_splits):
        for j, cs in enumerate(col_splits):
            patch = lab[rs[0] : rs[-1] + 1, cs[0] : cs[-1] + 1]
            meds[i, j] = np.median(patch.reshape(-1, 3), axis=0)
    return meds


def analyze(path: str, grid: tuple[int, int] = GRID, max_side: int = MAX_SIDE) -> ColorReport:
    rgb = load_rgb(path, max_side=max_side)
    lab = rgb2lab(rgb)
    means = patch_medians(lab, grid)
    rows, cols = grid

    # Coarse quadrants for the whole-panel color-difference metric: a knot or
    # hole covers a small fraction of a quadrant, so its median survives even
    # where a fine-grid patch median would not.
    quads = patch_medians(lab, COARSE_GRID).reshape(-1, 3)
    de = deltaE_ciede2000(quads[:, None, :], quads[None, :, :])
    color_diff_max_de = float(de.max())

    h, w, _ = lab.shape
    half = w // 2
    left = np.median(lab[:, :half].reshape(-1, 3), axis=0)
    right = np.median(lab[:, half:].reshape(-1, 3), axis=0)
    seam_de = float(deltaE_ciede2000(left[None, :], right[None, :])[0])

    L = means[..., 0]
    chroma = np.sqrt(means[..., 1] ** 2 + means[..., 2] ** 2)
    med_L, std_L = np.median(L), L.std()
    med_C, std_C = np.median(chroma), chroma.std()
    light_mask = L > med_L + 0.8 * std_L
    if std_C > 0:
        light_mask &= chroma < med_C - 0.3 * std_C
    sapwood_fraction = float(light_mask.mean())

    edge_span = max(1, cols // 4)
    col_hits = light_mask.any(axis=0)
    n_hit_cols = col_hits.sum()
    edge_hit_cols = col_hits[:edge_span].sum() + col_hits[-edge_span:].sum()
    sapwood_edge_fraction = float(edge_hit_cols / n_hit_cols) if n_hit_cols else 0.0

    flags = []
    if color_diff_max_de > COLOR_DIFF_DE_THRESHOLD:
        flags.append("barevny_rozdil")
    if seam_de > SEAM_DE_THRESHOLD:
        flags.append("rozbarvenost")
    if (
        sapwood_fraction > SAPWOOD_FRACTION_THRESHOLD
        and sapwood_edge_fraction > SAPWOOD_EDGE_CONCENTRATION_THRESHOLD
    ):
        flags.append("bel")

    return ColorReport(
        path=path,
        color_diff_max_de=round(color_diff_max_de, 2),
        seam_de=round(seam_de, 2),
        sapwood_fraction=round(sapwood_fraction, 3),
        sapwood_edge_fraction=round(sapwood_edge_fraction, 3),
        flags=flags,
    )


def resolve_images(source: str) -> list[str]:
    p = Path(source)
    if p.is_dir():
        return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    return [source]


def print_report(r: ColorReport) -> None:
    print(r.path)
    print(f"  color_diff_max_de       {r.color_diff_max_de}")
    print(f"  seam_de                 {r.seam_de}")
    print(f"  sapwood_fraction        {r.sapwood_fraction}")
    print(f"  sapwood_edge_fraction   {r.sapwood_edge_fraction}")
    print(f"  flags                   {r.flags or ['none']}")


def calibrate() -> None:
    files = sorted(glob.glob(str(REPO_ROOT / "data" / "set*" / "*.jpg")))
    print(f"{'file':<55s} {'de_max':>7s} {'seam_de':>8s} {'sap_frac':>9s} {'sap_edge':>9s}")
    for f in files:
        r = analyze(f)
        name = os.path.relpath(f, REPO_ROOT)
        print(
            f"{name:<55s} {r.color_diff_max_de:7.2f} {r.seam_de:8.2f} "
            f"{r.sapwood_fraction:9.3f} {r.sapwood_edge_fraction:9.3f}  {r.flags}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", nargs="?", help="image file or directory")
    p.add_argument("--calibrate", action="store_true", help="dump metrics for all of data/")
    args = p.parse_args()

    if args.calibrate:
        calibrate()
        return
    if not args.source:
        raise SystemExit("pass an image/folder, or --calibrate")

    for f in resolve_images(args.source):
        print_report(analyze(f))


if __name__ == "__main__":
    main()
