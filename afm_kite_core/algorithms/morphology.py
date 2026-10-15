"""Small binary morphology helpers."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def remove_objects_smaller_than(binary: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected objects smaller than a target area."""

    foreground = np.asarray(binary, dtype=bool)
    threshold = max(int(min_size), 1)
    labels, _ = ndi.label(
        foreground,
        structure=ndi.generate_binary_structure(foreground.ndim, 1),
    )
    counts = np.bincount(labels.ravel())
    keep = counts >= threshold
    keep[0] = False
    return keep[labels]
