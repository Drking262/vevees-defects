"""Texture/orientation feature extraction shared by grain_pattern.py.

No training data needed here either: these are classical structure-tensor
and Local Binary Pattern descriptors computed directly from the image.

Structure tensor gives, per pixel, the dominant local gradient direction and
how strongly oriented (vs isotropic) the neighborhood is. Averaging the
tensor components (not the angles -- angles wrap around, tensors don't) over
the whole image gives a global dominant orientation + coherence; averaging
over a grid of blocks and looking at how much the per-block orientation
varies gives a sense of how "wavy" the grain is locally.

In these portrait photos the grain runs roughly top-to-bottom (vertical), so
the intensity gradient (perpendicular to the grain stripes) is roughly
horizontal. `runout_deg` reports how far the dominant grain direction
deviates from vertical -- i.e. grain runout -- directly in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

from pipeline.imageutil import load_rgb

MAX_SIDE = 800
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
TENSOR_SIGMA = 4.0
BLOCK_GRID = (8, 8)


def _structure_tensor_components(gray: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gy, gx = np.gradient(gray)
    jxx = ndi.gaussian_filter(gx * gx, sigma)
    jyy = ndi.gaussian_filter(gy * gy, sigma)
    jxy = ndi.gaussian_filter(gx * gy, sigma)
    return jxx, jyy, jxy


def _orientation_and_coherence(jxx: float, jyy: float, jxy: float) -> tuple[float, float]:
    """theta in degrees (0-180, mod 180 since orientation is undirected), coherence in 0-1."""
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    theta_deg = np.degrees(theta) % 180
    denom = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / denom if denom > 1e-9 else 0.0
    return float(theta_deg), float(np.clip(coherence, 0.0, 1.0))


def _circular_std_deg(angles_deg: np.ndarray, period: float = 180.0) -> float:
    """Circular std of undirected angles (mod `period`), in degrees."""
    theta = np.radians(angles_deg) * (360.0 / period)
    mean_vec = np.mean(np.exp(1j * theta))
    r = np.abs(mean_vec)
    r = np.clip(r, 1e-9, 1.0)
    return float(np.degrees(np.sqrt(-2 * np.log(r))) / (360.0 / period))


def _angular_distance_deg(a: float, b: float, period: float = 180.0) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


@dataclass
class GrainFeatures:
    path: str
    lbp_hist: np.ndarray  # normalized histogram, length LBP_POINTS + 2
    global_coherence: float
    dominant_angle_deg: float  # gradient direction, 0-180
    grain_angle_deg: float  # dominant_angle_deg + 90 mod 180 (the stripe/grain direction)
    runout_deg: float  # deviation of grain_angle_deg from vertical (90 deg)
    orientation_std_deg: float  # spatial variability of local grain angle across the panel

    def vector(self) -> np.ndarray:
        """Concatenated, roughly-normalized feature vector for nearest-neighbor matching."""
        return np.concatenate(
            [
                self.lbp_hist,
                [
                    self.global_coherence,
                    self.runout_deg / 90.0,
                    self.orientation_std_deg / 90.0,
                ],
            ]
        )


def extract(path: str, max_side: int = MAX_SIDE) -> GrainFeatures:
    rgb = load_rgb(path, max_side=max_side)
    gray = rgb2gray(rgb)

    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    jxx, jyy, jxy = _structure_tensor_components(gray, TENSOR_SIGMA)
    dominant_angle_deg, global_coherence = _orientation_and_coherence(
        jxx.mean(), jyy.mean(), jxy.mean()
    )
    grain_angle_deg = (dominant_angle_deg + 90) % 180
    runout_deg = _angular_distance_deg(grain_angle_deg, 90.0)

    rows, cols = BLOCK_GRID
    h, w = gray.shape
    row_splits = np.array_split(np.arange(h), rows)
    col_splits = np.array_split(np.arange(w), cols)
    block_angles = []
    for rs in row_splits:
        for cs in col_splits:
            bxx = jxx[rs[0] : rs[-1] + 1, cs[0] : cs[-1] + 1].mean()
            byy = jyy[rs[0] : rs[-1] + 1, cs[0] : cs[-1] + 1].mean()
            bxy = jxy[rs[0] : rs[-1] + 1, cs[0] : cs[-1] + 1].mean()
            angle, coh = _orientation_and_coherence(bxx, byy, bxy)
            if coh > 0.1:  # skip near-isotropic blocks, their angle is meaningless noise
                block_angles.append((angle + 90) % 180)
    orientation_std_deg = _circular_std_deg(np.array(block_angles)) if block_angles else 90.0

    return GrainFeatures(
        path=path,
        lbp_hist=hist,
        global_coherence=round(global_coherence, 4),
        dominant_angle_deg=round(dominant_angle_deg, 1),
        grain_angle_deg=round(grain_angle_deg, 1),
        runout_deg=round(runout_deg, 1),
        orientation_std_deg=round(orientation_std_deg, 1),
    )
