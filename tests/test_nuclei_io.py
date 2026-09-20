from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nuclei_io import build_dataset_manifest, build_mask_cache, load_cached_mask, load_rgb


def make_sample(root: Path, sample_id: str = "sample") -> Path:
    sample = root / sample_id
    (sample / "images").mkdir(parents=True)
    (sample / "masks").mkdir()
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:12, 4:12] = 120
    Image.fromarray(image).save(sample / "images" / f"{sample_id}.png")
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[3:7, 3:7] = 255
    Image.fromarray(mask).save(sample / "masks" / "one.png")
    return sample


def test_16_bit_image_is_scaled_instead_of_saturated(tmp_path):
    path = tmp_path / "image.tiff"
    source = np.asarray([[0, 1024, 32768, 65535]], dtype=np.uint16)
    Image.fromarray(source).save(path)
    loaded = load_rgb(path)
    assert loaded.dtype == np.uint8
    assert loaded[0, 0, 0] == 0
    assert 0 < loaded[0, 1, 0] < 255
    assert loaded[0, -1, 0] == 255


def test_multiframe_tiff_is_rejected(tmp_path):
    path = tmp_path / "stack.tiff"
    first = Image.fromarray(np.zeros((8, 8), dtype=np.uint8))
    second = Image.fromarray(np.ones((8, 8), dtype=np.uint8))
    first.save(path, save_all=True, append_images=[second])
    with pytest.raises(ValueError, match="contains 2 frames"):
        load_rgb(path)


def test_cache_rebuilds_when_same_size_annotation_changes(tmp_path):
    sample = make_sample(tmp_path / "data")
    cache = tmp_path / "cache"
    first_manifest = build_dataset_manifest([sample])
    build_mask_cache([sample], cache, first_manifest)
    first = load_cached_mask(cache, sample.name).copy()
    changed = np.zeros((16, 16), dtype=np.uint8)
    changed[9:13, 9:13] = 255
    Image.fromarray(changed).save(sample / "masks" / "one.png")
    second_manifest = build_dataset_manifest([sample])
    build_mask_cache([sample], cache, second_manifest)
    second = load_cached_mask(cache, sample.name)
    assert first_manifest["dataset_fingerprint"] != second_manifest["dataset_fingerprint"]
    assert not np.array_equal(first, second)


def test_duplicate_instance_annotations_are_rejected(tmp_path):
    sample = make_sample(tmp_path / "data")
    original = sample / "masks" / "one.png"
    Image.open(original).save(sample / "masks" / "duplicate.png")
    manifest = build_dataset_manifest([sample])
    with pytest.raises(ValueError, match="Duplicate instance"):
        build_mask_cache([sample], tmp_path / "cache", manifest)
