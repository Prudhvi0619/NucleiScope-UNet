"""Post-processing helpers for nucleus instance separation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed


def remove_small_components(binary_mask: np.ndarray, minimum_size: int) -> np.ndarray:
    """Remove foreground components smaller than ``minimum_size`` pixels."""
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
    binary_mask = np.asarray(binary_mask, dtype=bool)
    components = label(binary_mask)
    if components.max() == 0:
        return components.astype(np.int32)

    distance = ndi.distance_transform_edt(binary_mask)
    coordinates = peak_local_max(
        distance,
        min_distance=minimum_distance,
        threshold_rel=threshold_relative,
        labels=binary_mask,
        exclude_border=False,
    )
    markers = np.zeros(binary_mask.shape, dtype=np.int32)
    for marker_id, (row, column) in enumerate(coordinates, start=1):
        markers[row, column] = marker_id

    # Every disconnected foreground region must contain at least one marker.
    next_marker = int(markers.max()) + 1
    for component_id in range(1, int(components.max()) + 1):
        component = components == component_id
        if np.any(markers[component] > 0):
            continue
        component_distance = np.where(component, distance, -1.0)
        row, column = np.unravel_index(
            np.argmax(component_distance), component_distance.shape
        )
        markers[row, column] = next_marker
        next_marker += 1

    return watershed(-distance, markers, mask=binary_mask).astype(np.int32)
