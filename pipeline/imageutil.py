"""Shared image loading helper for the classical-CV anomaly tools."""

from __future__ import annotations

import numpy as np
from PIL import Image


def load_rgb(path: str, max_side: int = 1024) -> np.ndarray:
    """Load an image as an RGB uint8 array, downsized so the longest side is max_side."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        return np.asarray(im)
