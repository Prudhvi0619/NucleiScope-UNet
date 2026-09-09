"""Run no-resize nuclei segmentation and quantification on one unseen image."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch

from postprocess import remove_small_components, separate_touching_nuclei
from train import UNet, load_rgb, predict_full_resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment and quantify nuclei without resizing the input image"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/best_model.pt")
    )
    parser.add_argument(
        "--test-summary",
        type=Path,
        default=Path("outputs/test/test_summary.json"),
        help="Uses validation-calibrated threshold and watershed settings when present.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/predictions")
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--minimum-size", type=int, default=None)
    parser.add_argument("--minimum-distance", type=int, default=None)
    parser.add_argument("--threshold-relative", type=float, default=None)
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    return parser.parse_args()


def load_evaluation_settings(summary_path: Path) -> dict:
    defaults = {
        "threshold": 0.5,
        "minimum_size": 5,
        "minimum_distance": 8,
        "threshold_relative": 0.05,
        "source": "built-in defaults",
    }
    if not summary_path.is_file():
        return defaults

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    selected = summary.get("counting", {}).get("selected_parameters", {})
    protocol = summary.get("protocol", {})
    return {
        "threshold": float(protocol.get("semantic_threshold", defaults["threshold"])),
        "minimum_size": int(
            selected.get("minimum_size", defaults["minimum_size"])
        ),
        "minimum_distance": int(
            selected.get("minimum_distance", defaults["minimum_distance"])
        ),
        "threshold_relative": float(
            selected.get("threshold_relative", defaults["threshold_relative"])
        ),
        "source": str(summary_path.resolve()),
    }


def validate_settings(settings: dict, patch_size: int, overlap: int) -> None:
    if not 0 < settings["threshold"] < 1:
        raise ValueError("Semantic threshold must be between 0 and 1")
    if settings["minimum_size"] < 1:
        raise ValueError("Minimum component size must be positive")
    if settings["minimum_distance"] < 1:
        raise ValueError("Watershed minimum distance must be positive")
    if not 0 <= settings["threshold_relative"] <= 1:
        raise ValueError("Watershed relative threshold must be between 0 and 1")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("Tile overlap must be between 0 and patch-size - 1")


def segmentation_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    color = np.asarray([255, 65, 20], dtype=np.float32)
    overlay[mask] = 0.58 * overlay[mask] + 0.42 * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def colorize_instances(labels: np.ndarray) -> np.ndarray:
    number_of_instances = int(labels.max())
    colors = np.zeros((number_of_instances + 1, 3), dtype=np.uint8)
    if number_of_instances:
        generator = np.random.default_rng(42)
        colors[1:] = generator.integers(
            low=45, high=256, size=(number_of_instances, 3), dtype=np.uint8
        )
    return colors[labels]


def instance_statistics(labels: np.ndarray, image_pixels: int) -> dict:
    number_of_instances = int(labels.max())
    areas = np.bincount(labels.ravel())[1:].astype(np.float64)
    if areas.size:
        diameters = np.sqrt(4.0 * areas / math.pi)
        area_summary = {
            "mean_area_pixels": float(areas.mean()),
            "median_area_pixels": float(np.median(areas)),
            "minimum_area_pixels": int(areas.min()),
            "maximum_area_pixels": int(areas.max()),
            "mean_equivalent_diameter_pixels": float(diameters.mean()),
            "median_equivalent_diameter_pixels": float(np.median(diameters)),
        }
    else:
        area_summary = {
            "mean_area_pixels": 0.0,
            "median_area_pixels": 0.0,
            "minimum_area_pixels": 0,
            "maximum_area_pixels": 0,
            "mean_equivalent_diameter_pixels": 0.0,
            "median_equivalent_diameter_pixels": 0.0,
        }
    return {
        "watershed_nucleus_count": number_of_instances,
        "nuclei_per_megapixel": float(number_of_instances * 1_000_000 / image_pixels),
        **area_summary,
    }


def save_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    instance_colors: np.ndarray,
    overlay: np.ndarray,
    destination: Path,
    nucleus_count: int,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Semantic mask")
    axes[2].imshow(instance_colors)
    axes[2].set_title(f"Watershed instances\nCount = {nucleus_count}")
    axes[3].imshow(overlay)
    axes[3].set_title("Segmentation overlay")
    for axis in axes:
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
    if args.tile_batch_size < 1:
        raise ValueError("Tile batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for practical full-resolution prediction")

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not config or config.get("resizing") is not False:
        raise ValueError("Checkpoint is not from the no-resize training pipeline")

    patch_size = int(config["patch_size"])
    overlap = (
        int(args.tile_overlap)
        if args.tile_overlap is not None
        else int(config["tile_overlap"])
    )
    settings = load_evaluation_settings(args.test_summary)
    if args.threshold is not None:
        settings["threshold"] = float(args.threshold)
    if args.minimum_size is not None:
        settings["minimum_size"] = int(args.minimum_size)
    if args.minimum_distance is not None:
        settings["minimum_distance"] = int(args.minimum_distance)
    if args.threshold_relative is not None:
        settings["threshold_relative"] = float(args.threshold_relative)
    validate_settings(settings, patch_size, overlap)

    model = UNet(base_channels=int(config["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image = load_rgb(args.image)
    original_height, original_width = image.shape[:2]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Image: {args.image.resolve()}")
    print(f"Original size: {original_width}x{original_height}")
    print("Geometric resizing: DISABLED")
    start_time = time.time()
    probability = predict_full_resolution(
        model,
        image,
        device,
        patch_size,
        overlap,
        args.tile_batch_size,
    )
    raw_mask = probability >= settings["threshold"]
    cleaned_mask = remove_small_components(raw_mask, settings["minimum_size"])
    connected_component_count = int(label_components(cleaned_mask))
    instance_labels = separate_touching_nuclei(
        cleaned_mask,
        minimum_distance=settings["minimum_distance"],
        threshold_relative=settings["threshold_relative"],
    )
    elapsed_seconds = time.time() - start_time

    image_output_dir = args.output_dir / args.image.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)
    probability_uint16 = np.round(np.clip(probability, 0, 1) * 65535).astype(
        np.uint16
    )
    Image.fromarray(probability_uint16).save(image_output_dir / "probability_map.png")
    Image.fromarray(raw_mask.astype(np.uint8) * 255).save(
        image_output_dir / "semantic_mask_raw.png"
    )
    Image.fromarray(cleaned_mask.astype(np.uint8) * 255).save(
        image_output_dir / "semantic_mask_cleaned.png"
    )
    Image.fromarray(instance_labels.astype(np.uint16)).save(
        image_output_dir / "instance_labels.tiff"
    )
    instance_colors = colorize_instances(instance_labels)
    Image.fromarray(instance_colors).save(image_output_dir / "instance_colors.png")
    overlay = segmentation_overlay(image, cleaned_mask)
    Image.fromarray(overlay).save(image_output_dir / "segmentation_overlay.png")

    image_pixels = original_height * original_width
    segmented_pixels = int(cleaned_mask.sum())
    quantification = {
        "input": {
            "image": str(args.image.resolve()),
            "width_pixels": original_width,
            "height_pixels": original_height,
            "source_image_resized": False,
        },
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "patch_size": patch_size,
            "tile_overlap": overlap,
        },
        "postprocessing": settings,
        "quantification": {
            "segmented_foreground_pixels": segmented_pixels,
            "foreground_coverage_percent": float(
                segmented_pixels * 100.0 / image_pixels
            ),
            "connected_component_count": connected_component_count,
            **instance_statistics(instance_labels, image_pixels),
        },
        "runtime_seconds": elapsed_seconds,
        "measurement_note": (
            "Areas and diameters are in pixels. Physical units require microscope "
            "pixel-size calibration. Nuclei per megapixel is an image-space density, "
            "not a physical density."
        ),
    }
    with (image_output_dir / "quantification.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(quantification, file, indent=2)

    save_comparison(
        image,
        cleaned_mask,
        instance_colors,
        overlay,
        image_output_dir / "prediction_summary.png",
        int(instance_labels.max()),
    )

    print("\nPrediction complete")
    print(f"Connected components: {connected_component_count}")
    print(f"Watershed nucleus count: {int(instance_labels.max())}")
    print(f"Foreground coverage: {quantification['quantification']['foreground_coverage_percent']:.2f}%")
    print(f"Runtime: {elapsed_seconds:.2f} seconds")
    print(f"Outputs: {image_output_dir.resolve()}")


def label_components(binary_mask: np.ndarray) -> int:
    """Return the number of 8-connected foreground components."""
    from scipy import ndimage as ndi

    _, number_of_components = ndi.label(
        binary_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    return int(number_of_components)


if __name__ == "__main__":
    main()
