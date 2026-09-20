"""Post-processing helpers for nucleus instance separation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed


def remove_small_components(binary_mask: np.ndarray, minimum_size: int) -> np.ndarray:
    """Remove foreground components smaller than ``minimum_size`` pixels."""
    if minimum_size < 1:
        raise ValueError("minimum_size must be positive")
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if minimum_size <= 1:
        return binary_mask
    components = label(binary_mask)
    component_sizes = np.bincount(components.ravel())
    keep = component_sizes >= minimum_size
    keep[0] = False
    return keep[components]


def separate_touching_nuclei(
    binary_mask: np.ndarray,
    minimum_distance: int,
    threshold_relative: float,
) -> np.ndarray:
    """Split touching foreground regions using marker-controlled watershed."""
    if minimum_distance < 1:
        raise ValueError("minimum_distance must be positive")
    if not 0 <= threshold_relative <= 1:
        raise ValueError("threshold_relative must be between 0 and 1")
    binary_mask = np.asarray(binary_mask, dtype=bool)
    components = label(binary_mask)
    if components.max() == 0:
        return components.astype(np.int32)

    distance = ndi.distance_transform_edt(binary_mask)
    markers = np.zeros(binary_mask.shape, dtype=np.int32)
    next_marker = 1
    # Search per component so a large object cannot suppress peaks in smaller ones
    # through peak_local_max's image-global relative threshold.
    for component_id in range(1, int(components.max()) + 1):
        component = components == component_id
        coordinates = peak_local_max(
            distance,
            min_distance=minimum_distance,
            threshold_rel=threshold_relative,
            labels=component,
            exclude_border=False,
        )
        if len(coordinates) == 0:
            component_distance = np.where(component, distance, -1.0)
            coordinates = np.asarray(
                [np.unravel_index(np.argmax(component_distance), component_distance.shape)]
            )
        for row, column in coordinates:
            markers[int(row), int(column)] = next_marker
            next_marker += 1

    return watershed(-distance, markers, mask=binary_mask).astype(np.int32)
