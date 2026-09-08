"""Run NucleiScope on one image and produce basic measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries, watershed

import torch

from train import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict nuclei in one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/best_model.pt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prediction"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--minimum-size",
        type=int,
        default=20,
        help="Remove predicted components smaller than this many model pixels.",
    )
    parser.add_argument(
        "--peak-min-distance",
        type=int,
        default=5,
        help="Minimum distance between watershed nucleus centres.",
    )
    parser.add_argument(
        "--peak-threshold-relative",
        type=float,
        default=0.05,
        help="Relative distance-map threshold for watershed markers.",
    )
    return parser.parse_args()


def separate_touching_nuclei(
    binary_mask: np.ndarray,
    minimum_distance: int,
    threshold_relative: float,
) -> np.ndarray:
    """Split touching foreground regions using marker-controlled watershed."""
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

    # Guarantee at least one marker inside every disconnected foreground region.
    components = label(binary_mask)
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

    if markers.max() == 0:
        return components
    return watershed(-distance, markers, mask=binary_mask).astype(np.int32)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.peak_min_distance < 1:
        raise ValueError("--peak-min-distance must be at least 1")
    if not 0.0 <= args.peak_threshold_relative < 1.0:
        raise ValueError("--peak-threshold-relative must be in [0, 1)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 256))
    base_channels = int(checkpoint.get("base_channels", 32))

    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    original = Image.open(args.image).convert("RGB")
    resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        logits = model(tensor)
    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    binary_mask = probability >= args.threshold
    cleaned_mask = remove_small_objects(binary_mask, min_size=args.minimum_size)
    connected_component_count = int(label(cleaned_mask).max())
    labelled_mask = separate_touching_nuclei(
        cleaned_mask,
        minimum_distance=args.peak_min_distance,
        threshold_relative=args.peak_threshold_relative,
    )
    regions = regionprops(labelled_mask)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_mask = Image.fromarray(cleaned_mask.astype(np.uint8) * 255)
    output_mask = model_mask.resize(original.size, Image.Resampling.NEAREST)
    output_mask.save(args.output_dir / "prediction_mask.png")

    coloured_instances = label2rgb(
        labelled_mask, bg_label=0, bg_color=(0, 0, 0), kind="overlay"
    )
    instance_image = Image.fromarray(
        (np.clip(coloured_instances, 0, 1) * 255).astype(np.uint8)
    ).resize(original.size, Image.Resampling.NEAREST)
    instance_image.save(args.output_dir / "instance_labels.png")

    boundaries = find_boundaries(labelled_mask, mode="outer")
    boundaries_image = Image.fromarray(boundaries.astype(np.uint8) * 255).resize(
        original.size, Image.Resampling.NEAREST
    )
    boundaries_array = np.asarray(boundaries_image) > 0
    overlay = np.asarray(original, dtype=np.uint8).copy()
    overlay[boundaries_array] = np.array([255, 40, 40], dtype=np.uint8)
    Image.fromarray(overlay).save(args.output_dir / "prediction_overlay.png")

    measurement_rows = []
    for region in regions:
        measurement_rows.append(
            {
                "nucleus_id": region.label,
                "area_model_pixels": float(region.area),
                "perimeter_model_pixels": float(region.perimeter),
                "centroid_row": float(region.centroid[0]),
                "centroid_column": float(region.centroid[1]),
            }
        )

    measurement_fields = [
        "nucleus_id",
        "area_model_pixels",
        "perimeter_model_pixels",
        "centroid_row",
        "centroid_column",
    ]
    with (args.output_dir / "nuclei_measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=measurement_fields)
        writer.writeheader()
        writer.writerows(measurement_rows)

    areas = [float(region.area) for region in regions]
    summary = {
        "input_image": str(args.image.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "threshold": args.threshold,
        "minimum_component_size": args.minimum_size,
        "separation_method": "marker-controlled watershed",
        "watershed_peak_min_distance": args.peak_min_distance,
        "watershed_peak_threshold_relative": args.peak_threshold_relative,
        "connected_component_count_before_watershed": connected_component_count,
        "nucleus_count": len(regions),
        "nucleus_density_percent": float(cleaned_mask.mean() * 100.0),
        "mean_nucleus_area_model_pixels": float(np.mean(areas)) if areas else 0.0,
        "median_nucleus_area_model_pixels": float(np.median(areas)) if areas else 0.0,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Prediction outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
