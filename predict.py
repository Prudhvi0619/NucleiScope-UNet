"""Run provenance-checked nuclei segmentation and quantification on one image."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import label

from nuclei_io import atomic_json_dump, load_rgb, save_instance_labels, sha256_file
from postprocess import remove_small_components, separate_touching_nuclei
from train import UNet, load_checkpoint, predict_full_resolution, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict nuclei without resizing the source image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/run/best_model.pt"))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="calibration.json produced by test.py for this exact checkpoint",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--connected-minimum-size", type=int, default=None)
    parser.add_argument("--watershed-minimum-size", type=int, default=None)
    parser.add_argument("--minimum-distance", type=int, default=None)
    parser.add_argument("--threshold-relative", type=float, default=None)
    parser.add_argument(
        "--postprocess-scale-factor",
        type=float,
        default=1.0,
        help="Linear pixel-scale ratio versus calibration; areas scale by its square.",
    )
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def load_settings(args: argparse.Namespace, checkpoint_sha: str) -> tuple[dict, dict]:
    settings: dict[str, float | int] = {}
    sources: dict[str, str] = {}
    if args.calibration is not None:
        if not args.calibration.is_file():
            raise FileNotFoundError(f"Calibration file not found: {args.calibration}")
        with args.calibration.open("r", encoding="utf-8") as file:
            calibration = json.load(file)
        if calibration.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Calibration file belongs to a different checkpoint")
        settings["threshold"] = float(calibration["semantic_threshold"])
        settings.update(calibration["counting"])
        sources = {key: f"calibration:{args.calibration.name}" for key in settings}
    overrides = {
        "threshold": args.threshold,
        "connected_minimum_size": args.connected_minimum_size,
        "watershed_minimum_size": args.watershed_minimum_size,
        "minimum_distance": args.minimum_distance,
        "threshold_relative": args.threshold_relative,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
            sources[key] = "command-line override"
    required = set(overrides)
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(
            "No unverified post-processing defaults are used. Provide --calibration or "
            f"all manual settings; missing: {', '.join(missing)}"
        )
    return settings, sources


def validate_settings(settings: dict, patch_size: int, overlap: int, tile_batch_size: int) -> None:
    if not 0 < float(settings["threshold"]) < 1:
        raise ValueError("Semantic threshold must be between 0 and 1")
    for key in ("connected_minimum_size", "watershed_minimum_size", "minimum_distance"):
        if int(settings[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= float(settings["threshold_relative"]) <= 1:
        raise ValueError("threshold_relative must be between 0 and 1")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("Tile overlap must be between 0 and patch-size - 1")
    if tile_batch_size < 1:
        raise ValueError("Tile batch size must be positive")


def segmentation_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    overlay[mask] = 0.58 * overlay[mask] + 0.42 * np.asarray([255, 65, 20], np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def colorize_instances(labels: np.ndarray) -> np.ndarray:
    number = int(labels.max(initial=0))
    colors = np.zeros((number + 1, 3), dtype=np.uint8)
    if number:
        colors[1:] = np.random.default_rng(42).integers(45, 256, size=(number, 3), dtype=np.uint8)
    return colors[labels]


def instance_statistics(labels: np.ndarray, image_pixels: int) -> dict:
    number = int(labels.max(initial=0))
    areas = np.bincount(labels.ravel())[1:].astype(np.float64)
    diameters = np.sqrt(4.0 * areas / math.pi) if areas.size else np.asarray([], dtype=np.float64)
    return {
        "watershed_nucleus_count": number,
        "nuclei_per_megapixel": float(number * 1_000_000 / image_pixels),
        "mean_area_pixels": float(areas.mean()) if areas.size else 0.0,
        "median_area_pixels": float(np.median(areas)) if areas.size else 0.0,
        "minimum_area_pixels": int(areas.min()) if areas.size else 0,
        "maximum_area_pixels": int(areas.max()) if areas.size else 0,
        "mean_equivalent_diameter_pixels": float(diameters.mean()) if areas.size else 0.0,
        "median_equivalent_diameter_pixels": float(np.median(diameters)) if areas.size else 0.0,
    }


def save_comparison(image, mask, colors, overlay, destination, count):
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for axis, panel, title in zip(
        axes,
        (image, mask, colors, overlay),
        (
            "Original image",
            "Semantic mask",
            f"Watershed instances\nCount = {count}",
            "Segmentation overlay",
        ),
        strict=False,
    ):
        axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.postprocess_scale_factor <= 0:
        raise ValueError("--postprocess-scale-factor must be positive")
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_sha = sha256_file(args.checkpoint)
    contract = checkpoint["run_contract"]
    patch_size = int(contract["patch_size"])
    overlap = (
        int(args.tile_overlap) if args.tile_overlap is not None else int(contract["tile_overlap"])
    )
    settings, setting_sources = load_settings(args, checkpoint_sha)
    scale = float(args.postprocess_scale_factor)
    if not math.isclose(scale, 1.0):
        for key in ("connected_minimum_size", "watershed_minimum_size"):
            settings[key] = max(1, int(round(float(settings[key]) * scale * scale)))
            setting_sources[key] += f"; scaled by {scale:g}^2"
        settings["minimum_distance"] = max(
            1, int(round(float(settings["minimum_distance"]) * scale))
        )
        setting_sources["minimum_distance"] += f"; scaled by {scale:g}"
    validate_settings(settings, patch_size, overlap, args.tile_batch_size)

    model = UNet(int(contract["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image = load_rgb(args.image)
    height, width = image.shape[:2]
    start = time.time()
    probability = predict_full_resolution(
        model, image, device, patch_size, overlap, args.tile_batch_size
    )
    raw_mask = probability >= float(settings["threshold"])
    connected_mask = remove_small_components(raw_mask, int(settings["connected_minimum_size"]))
    connected_count = int(label(connected_mask).max())
    watershed_mask = remove_small_components(raw_mask, int(settings["watershed_minimum_size"]))
    instances = separate_touching_nuclei(
        watershed_mask, int(settings["minimum_distance"]), float(settings["threshold_relative"])
    )
    elapsed = time.time() - start

    image_sha = sha256_file(args.image)
    image_output_dir = args.output_dir / f"{args.image.stem}-{image_sha[:12]}"
    if image_output_dir.exists() and any(image_output_dir.iterdir()):
        raise FileExistsError(f"Prediction output already exists: {image_output_dir}")
    image_output_dir.mkdir(parents=True, exist_ok=True)
    probability_uint16 = np.round(np.clip(probability, 0, 1) * 65535).astype(np.uint16)
    Image.fromarray(probability_uint16).save(image_output_dir / "probability_map.png")
    Image.fromarray(raw_mask.astype(np.uint8) * 255).save(
        image_output_dir / "semantic_mask_raw.png"
    )
    Image.fromarray(watershed_mask.astype(np.uint8) * 255).save(
        image_output_dir / "semantic_mask_cleaned.png"
    )
    save_instance_labels(instances, image_output_dir / "instance_labels.tiff")
    colors = colorize_instances(instances)
    Image.fromarray(colors).save(image_output_dir / "instance_colors.png")
    overlay = segmentation_overlay(image, watershed_mask)
    Image.fromarray(overlay).save(image_output_dir / "segmentation_overlay.png")

    pixels, segmented = height * width, int(watershed_mask.sum())
    quantification = {
        "schema_version": 2,
        "input": {
            "file_name": args.image.name,
            "sha256": image_sha,
            "width_pixels": width,
            "height_pixels": height,
            "source_image_resized": False,
        },
        "model": {
            "checkpoint_name": args.checkpoint.name,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "patch_size": patch_size,
            "tile_overlap": overlap,
        },
        "postprocessing": {"values": settings, "sources": setting_sources, "scale_factor": scale},
        "quantification": {
            "segmented_foreground_pixels": segmented,
            "foreground_coverage_percent": float(segmented * 100.0 / pixels),
            "connected_component_count": connected_count,
            **instance_statistics(instances, pixels),
        },
        "runtime": {"device": str(device), "seconds": elapsed},
        "measurement_note": (
            "Areas, distances, and diameters are in pixels. Apply a verified physical "
            "pixel-size calibration before interpreting biological scale or density."
        ),
    }
    atomic_json_dump(quantification, image_output_dir / "quantification.json")
    save_comparison(
        image,
        watershed_mask,
        colors,
        overlay,
        image_output_dir / "prediction_summary.png",
        int(instances.max(initial=0)),
    )
    print(
        json.dumps(
            {
                "connected_components": connected_count,
                "watershed_nuclei": int(instances.max(initial=0)),
                "runtime_seconds": elapsed,
                "output": str(image_output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
