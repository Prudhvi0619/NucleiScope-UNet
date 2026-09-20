"""Validated image, annotation, cache, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CACHE_SCHEMA_VERSION = 2


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json_dump(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(destination)


def image_path_for(sample_dir: Path) -> Path:
    image_dir = sample_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    candidates = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one image in {image_dir}, found {len(candidates)}")
    return candidates[0]


def mask_paths_for(sample_dir: Path) -> list[Path]:
    mask_dir = sample_dir / "masks"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
    paths = sorted(
        path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No instance masks found in {mask_dir}")
    return paths


def collect_samples(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    samples = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and (path / "images").is_dir() and (path / "masks").is_dir()
    )
    if len(samples) < 10:
        raise ValueError(f"Expected at least 10 annotated samples, found {len(samples)}")
    return samples


def _ensure_single_frame(image: Image.Image, path: Path) -> None:
    frames = int(getattr(image, "n_frames", 1))
    if frames != 1:
        raise ValueError(
            f"{path} contains {frames} frames. NucleiScope expects a single 2-D "
            "image; export a projection or individual slice explicitly."
        )


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    channel = np.asarray(channel)
    if channel.dtype == np.uint8:
        return channel.copy()
    values = channel.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.all():
        values = np.where(finite, values, 0.0)
    # Robust per-image scaling handles both full-range 16-bit data and common
    # 10/12-bit acquisitions stored inside a uint16 container.
    low, high = np.percentile(values, (0.5, 99.5))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    values = (values - low) / (high - low)
    return np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_rgb(image_path: Path) -> np.ndarray:
    """Load one 2-D image without silently truncating bit depth or TIFF stacks."""
    with Image.open(image_path) as opened:
        _ensure_single_frame(opened, image_path)
        image = ImageOps.exif_transpose(opened)
        if image.mode in {"P", "CMYK", "YCbCr", "HSV", "RGBA", "LA"}:
            array = np.asarray(image.convert("RGB"))
        else:
            array = np.asarray(image)
    if array.ndim == 2:
        channel = _normalize_channel(array)
        return np.repeat(channel[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D image, got shape {array.shape} from {image_path}")
    if array.shape[2] == 1:
        channel = _normalize_channel(array[..., 0])
        return np.repeat(channel[..., None], 3, axis=2)
    if array.shape[2] < 3:
        raise ValueError(f"Unsupported channel layout {array.shape} in {image_path}")
    return np.stack([_normalize_channel(array[..., i]) for i in range(3)], axis=2)


def load_binary_mask(mask_path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(mask_path) as opened:
        _ensure_single_frame(opened, mask_path)
        array = np.asarray(opened)
    if array.ndim == 3:
        array = np.any(array != 0, axis=2)
    elif array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {array.shape}: {mask_path}")
    mask = np.asarray(array != 0, dtype=bool)
    if mask.shape != expected_shape:
        raise ValueError(f"Mask {mask_path} has shape {mask.shape}; expected {expected_shape}")
    return mask


def sample_manifest(sample_dir: Path) -> dict:
    image_path = image_path_for(sample_dir)
    masks = mask_paths_for(sample_dir)
    image_sha = sha256_file(image_path)
    mask_records = [{"name": p.name, "sha256": sha256_file(p)} for p in masks]
    annotation_fingerprint = sha256_json(mask_records)
    return {
        "sample_id": sample_dir.name,
        "image": {"name": image_path.name, "sha256": image_sha},
        "masks": mask_records,
        "annotation_fingerprint": annotation_fingerprint,
        "sample_fingerprint": sha256_json(
            {"image_sha256": image_sha, "annotation_fingerprint": annotation_fingerprint}
        ),
    }


def build_dataset_manifest(samples: Sequence[Path]) -> dict:
    records = [sample_manifest(sample) for sample in samples]
    return {
        "schema_version": 1,
        "sample_count": len(records),
        "samples": records,
        "dataset_fingerprint": sha256_json(records),
    }


def _cache_paths(cache_dir: Path, sample_id: str) -> tuple[Path, Path, Path]:
    return (
        cache_dir / f"{sample_id}.npy",
        cache_dir / f"{sample_id}.centers.npy",
        cache_dir / f"{sample_id}.json",
    )


def _atomic_save_array(array: np.ndarray, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(destination)


def _valid_cache(
    mask_path: Path,
    centers_path: Path,
    metadata_path: Path,
    manifest: dict,
    expected_shape: tuple[int, int],
) -> dict | None:
    if not (mask_path.is_file() and centers_path.is_file() and metadata_path.is_file()):
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        if metadata.get("annotation_fingerprint") != manifest["annotation_fingerprint"]:
            return None
        if tuple(metadata.get("shape", ())) != expected_shape:
            return None
        mask = np.load(mask_path, allow_pickle=False)
        centers = np.load(centers_path, allow_pickle=False)
        if mask.shape != expected_shape or mask.dtype != np.uint8:
            return None
        if centers.ndim != 2 or centers.shape[1:] != (2,):
            return None
        if int(metadata.get("instance_count", -1)) != len(centers):
            return None
        if not np.isin(mask, (0, 1)).all():
            return None
        return metadata
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def build_mask_cache(
    samples: Sequence[Path], cache_dir: Path, dataset_manifest: dict | None = None
) -> dict[str, dict]:
    """Create content-addressed semantic caches and validate every instance mask."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_by_id = (
        {row["sample_id"]: row for row in dataset_manifest["samples"]}
        if dataset_manifest is not None
        else {}
    )
    metadata_by_id: dict[str, dict] = {}
    structure = np.ones((3, 3), dtype=np.uint8)
    for sample_dir in samples:
        manifest = manifest_by_id.get(sample_dir.name) or sample_manifest(sample_dir)
        image = load_rgb(image_path_for(sample_dir))
        expected_shape = image.shape[:2]
        mask_path, centers_path, metadata_path = _cache_paths(cache_dir, sample_dir.name)
        cached = _valid_cache(mask_path, centers_path, metadata_path, manifest, expected_shape)
        if cached is not None:
            metadata_by_id[sample_dir.name] = cached
            continue

        combined = np.zeros(expected_shape, dtype=np.uint8)
        centers: list[tuple[float, float]] = []
        seen_hashes: set[str] = set()
        for record, annotation_path in zip(
            manifest["masks"], mask_paths_for(sample_dir), strict=False
        ):
            if record["sha256"] in seen_hashes:
                raise ValueError(f"Duplicate instance annotation detected: {annotation_path}")
            seen_hashes.add(record["sha256"])
            instance = load_binary_mask(annotation_path, expected_shape)
            if not instance.any():
                raise ValueError(f"Empty instance annotation: {annotation_path}")
            _, components = ndi.label(instance, structure=structure)
            if components != 1:
                raise ValueError(
                    f"Each annotation must contain exactly one connected nucleus; "
                    f"{annotation_path} contains {components} components"
                )
            if np.any((combined != 0) & instance):
                raise ValueError(f"Overlapping instance annotations detected in {sample_dir}")
            rows, columns = np.nonzero(instance)
            centers.append((float(rows.mean()), float(columns.mean())))
            combined[instance] = 1

        centers_array = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "sample_id": sample_dir.name,
            "shape": list(expected_shape),
            "annotation_fingerprint": manifest["annotation_fingerprint"],
            "sample_fingerprint": manifest["sample_fingerprint"],
            "instance_count": int(len(centers_array)),
            "foreground_pixels": int(combined.sum()),
            "foreground_fraction": float(combined.mean()),
        }
        _atomic_save_array(combined, mask_path)
        _atomic_save_array(centers_array, centers_path)
        atomic_json_dump(metadata, metadata_path)
        metadata_by_id[sample_dir.name] = metadata
    return metadata_by_id


def load_cached_mask(cache_dir: Path, sample_id: str) -> np.ndarray:
    mask = np.load(cache_dir / f"{sample_id}.npy", allow_pickle=False)
    return np.asarray(mask, dtype=np.uint8)


def load_cached_centers(cache_dir: Path, sample_id: str) -> np.ndarray:
    centers = np.load(cache_dir / f"{sample_id}.centers.npy", allow_pickle=False)
    return np.asarray(centers, dtype=np.float32)


def load_instance_labels(sample_dir: Path) -> np.ndarray:
    image = load_rgb(image_path_for(sample_dir))
    shape = image.shape[:2]
    labels = np.zeros(shape, dtype=np.int32)
    for instance_id, mask_path in enumerate(mask_paths_for(sample_dir), start=1):
        mask = load_binary_mask(mask_path, shape)
        if not mask.any():
            raise ValueError(f"Empty instance annotation: {mask_path}")
        if np.any(labels[mask] != 0):
            raise ValueError(f"Overlapping instance annotations detected in {sample_dir}")
        labels[mask] = instance_id
    return labels


def save_instance_labels(labels: np.ndarray, destination: Path) -> None:
    maximum = int(np.max(labels, initial=0))
    if maximum > np.iinfo(np.int32).max:
        raise OverflowError(f"Instance label {maximum} exceeds signed 32-bit TIFF capacity")
    Image.fromarray(np.asarray(labels, dtype=np.int32), mode="I").save(destination)


def ensure_new_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new directory to avoid "
            "mixing or overwriting experiment artifacts."
        )
    path.mkdir(parents=True, exist_ok=True)
