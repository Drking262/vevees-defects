"""Debug visualization for color_anomaly.py and grain_pattern.py: draws what
each metric actually measures on top of the photo, instead of just printing
numbers, so a threshold decision can be eyeballed against the image.

Color plot: the same 6x8 grid color_anomaly.py scores, sapwood-flagged
patches outlined in red, and the left/right seam split as a dashed line.

Grain plot: per-block dominant grain direction (from grain_features.py's
structure tensor) drawn as a tick per block, colored by how coherent
(confidently oriented) that block is, against a true-vertical reference line
-- so runout/waviness numbers can be checked against the actual tilt shown.

Usage:
    python -m pipeline.visualize path/to/image.jpg
    python -m pipeline.visualize path/to/folder/ --out viz_out
    python -m pipeline.visualize path/to/image.jpg --kind color
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray, rgb2lab

from pipeline.color_anomaly import GRID, analyze as color_analyze, patch_medians
from pipeline.grain_features import (
    BLOCK_GRID,
    MAX_SIDE as GRAIN_MAX_SIDE,
    TENSOR_SIGMA,
    _orientation_and_coherence,
    _structure_tensor_components,
    extract as grain_extract,
)
from pipeline.imageutil import load_rgb

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_OUT = REPO_ROOT / "viz_out"


def viz_color(path: str, out_path: Path) -> None:
    rgb = load_rgb(path, max_side=1024)
    lab = rgb2lab(rgb)
    report = color_analyze(path)

    rows, cols = GRID
    h, w = rgb.shape[:2]
    row_edges = [int(x[0]) for x in np.array_split(np.arange(h), rows)] + [h]
    col_edges = [int(x[0]) for x in np.array_split(np.arange(w), cols)] + [w]

    means = patch_medians(lab, GRID)
    L = means[..., 0]
    chroma = np.sqrt(means[..., 1] ** 2 + means[..., 2] ** 2)
    med_L, std_L = np.median(L), L.std()
    med_C, std_C = np.median(chroma), chroma.std()
    light_mask = L > med_L + 0.8 * std_L
    if std_C > 0:
        light_mask &= chroma < med_C - 0.3 * std_C

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.imshow(rgb)
    for i in range(rows):
        for j in range(cols):
            y0, y1 = row_edges[i], row_edges[i + 1]
            x0, x1 = col_edges[j], col_edges[j + 1]
            flagged = light_mask[i, j]
            color = "red" if flagged else "white"
            lw = 2.5 if flagged else 0.6
            alpha = 0.9 if flagged else 0.4
            rect = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=lw, alpha=alpha
            )
            ax.add_patch(rect)
    ax.axvline(w / 2, color="cyan", linewidth=2, linestyle="--")
    ax.set_title(
        f"{path}\nflags={report.flags or ['none']}  seam_dE={report.seam_de}  "
        f"max_dE={report.color_diff_max_de}  sapwood_frac={report.sapwood_fraction}",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def viz_grain(path: str, out_path: Path) -> None:
    rgb = load_rgb(path, max_side=GRAIN_MAX_SIDE)
    gray = rgb2gray(rgb)
    feat = grain_extract(path)

    jxx, jyy, jxy = _structure_tensor_components(gray, TENSOR_SIGMA)
    rows, cols = BLOCK_GRID
    h, w = gray.shape
    row_splits = np.array_split(np.arange(h), rows)
    col_splits = np.array_split(np.arange(w), cols)

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.imshow(rgb)
    for rs in row_splits:
        for cs in col_splits:
            bxx = jxx[rs[0] : rs[-1] + 1, cs[0] : cs[-1] + 1].mean()
            byy = jyy[rs[0] : rs[-1] + 1, cs[0] : cs[-1] + 1].mean()
            bxy = jxy[rs[0] : rs[-1] + 1, cs[0] : cs[-1] + 1].mean()
            angle, coh = _orientation_and_coherence(bxx, byy, bxy)
            if coh <= 0.1:
                continue
            grain_ang = (angle + 90) % 180
            cy = (rs[0] + rs[-1]) / 2
            cx = (cs[0] + cs[-1]) / 2
            length = min(h / rows, w / cols) * 0.45
            # grain_angle_deg follows the module's own convention: 0=horizontal,
            # 90=vertical, in the same (x=col, y=row) frame matplotlib draws in.
            theta = np.radians(grain_ang)
            dx = length * np.cos(theta)
            dy = length * np.sin(theta)
            color = plt.cm.viridis(min(coh, 1.0))
            ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, linewidth=2)

    ax.axvline(w / 2, color="red", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_title(
        f"{path}\ngrain_angle={feat.grain_angle_deg}°  runout={feat.runout_deg}°  "
        f"orientation_std={feat.orientation_std_deg}°  coherence={feat.global_coherence}",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def resolve_images(source: str) -> list[str]:
    p = Path(source)
    if p.is_dir():
        return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    return [source]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="image file or directory")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output directory for PNGs")
    p.add_argument("--kind", choices=["color", "grain", "both"], default="both")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in resolve_images(args.source):
        stem = Path(f).stem
        if args.kind in ("color", "both"):
            out_path = out_dir / f"{stem}_color.png"
            viz_color(f, out_path)
            print(f"wrote {out_path}")
        if args.kind in ("grain", "both"):
            out_path = out_dir / f"{stem}_grain.png"
            viz_grain(f, out_path)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
