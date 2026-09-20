import numpy as np
import pytest

from test import instance_metrics, segmentation_metrics
from train import pad_to_patch, tile_origins


def test_padding_does_not_reflect_unlabelled_content():
    image = np.full((2, 2, 3), 255, dtype=np.uint8)
    mask = np.ones((2, 2), dtype=np.uint8)
    padded_image, padded_mask = pad_to_patch(image, mask, 4)
    assert np.all(padded_image[2:] == 0)
    assert np.all(padded_mask[2:] == 0)


def test_tile_origins_cover_final_pixel():
    origins = tile_origins(701, 256, 64)
    assert origins[0] == 0
    assert origins[-1] == 701 - 256
    assert all(right - left <= 256 for left, right in zip(origins, origins[1:], strict=False))


def test_invalid_tile_configuration_is_rejected():
    with pytest.raises(ValueError):
        tile_origins(100, 64, 64)


def test_instance_metrics_perfect_match():
    truth = np.zeros((8, 8), dtype=np.int32)
    truth[1:3, 1:3] = 1
    truth[5:7, 5:7] = 2
    result = instance_metrics(truth, truth.copy())
    assert result["instance_f1_0.50"] == 1.0
    assert result["instance_mean_ap_0.50_0.95"] == 1.0


def test_instance_metrics_penalize_merged_nuclei():
    truth = np.zeros((8, 8), dtype=np.int32)
    truth[1:3, 1:3] = 1
    truth[1:3, 4:6] = 2
    prediction = np.zeros_like(truth)
    prediction[1:3, 1:6] = 1
    result = instance_metrics(truth, prediction)
    assert result["instance_f1_0.50"] < 1.0


def test_empty_semantic_pair_scores_as_correct():
    result = segmentation_metrics(np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4), bool), 0.5)
    assert result["dice"] == 1.0
    assert result["foreground_area_error_percent"] == 0.0
