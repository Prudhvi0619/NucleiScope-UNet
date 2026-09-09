"""Evaluate the no-resize U-Net on the untouched test split.

The script reports semantic-segmentation metrics at original image resolution
and nucleus-count metrics after validation-only watershed calibration. Test data
is never used to select the model, threshold, or post-processing parameters.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import label
from tqdm import tqdm

import torch

from postprocess import remove_small_components, separate_touching_nuclei
from train import (
    UNet,
    build_mask_cache,
    collect_samples,
    image_path_for,
    load_rgb,
    mask_paths_for,
    predict_full_resolution,
    seed_everything,
)


SEGMENTATION_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "pixel_accuracy",
    "mcc",
    "binary_cross_entropy",
    "foreground_area_error_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-resolution segmentation and counting evaluation"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/stage1_train"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/best_model.pt")
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=None,
        help="Defaults to splits.json beside the checkpoint.",
    )
    parser.add_argument(
        "--mask-cache",
        type=Path,
        default=None,
        help="Defaults to the original-resolution mask cache beside checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=None,
        help="Defaults to the overlap stored in the training checkpoint.",
    )
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(empty_value)


def segmentation_metrics(
    probability: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=bool)
    prediction = probability >= threshold
    true_positive = int(np.logical_and(prediction, target).sum())
    true_negative = int(np.logical_and(~prediction, ~target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())

    dice = safe_divide(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
        empty_value=1.0,
    )
    iou = safe_divide(
        true_positive,
        true_positive + false_positive + false_negative,
        empty_value=1.0,
    )
    precision = safe_divide(
        true_positive, true_positive + false_positive, empty_value=1.0
    )
    recall = safe_divide(
        true_positive, true_positive + false_negative, empty_value=1.0
    )
    specificity = safe_divide(
        true_negative, true_negative + false_positive, empty_value=1.0
    )
    pixel_accuracy = safe_divide(
        true_positive + true_negative,
        true_positive + true_negative + false_positive + false_negative,
    )
    mcc_denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = safe_divide(
        true_positive * true_negative - false_positive * false_negative,
        mcc_denominator,
    )
    clipped = np.clip(probability.astype(np.float64), 1e-7, 1 - 1e-7)
    binary_cross_entropy = float(
        -np.mean(target * np.log(clipped) + (~target) * np.log(1 - clipped))
    )
    foreground_area_error = safe_divide(
        abs(int(prediction.sum()) - int(target.sum())) * 100.0,
        int(target.sum()),
    )
    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "pixel_accuracy": pixel_accuracy,
        "mcc": mcc,
        "binary_cross_entropy": binary_cross_entropy,
        "foreground_area_error_percent": foreground_area_error,
    }


def aggregate_segmentation(rows: list[dict]) -> dict:
    macro = {}
    for metric in SEGMENTATION_METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        macro[metric] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    totals = {
        key: int(sum(row[key] for row in rows))
        for key in (
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
        )
    }
    tp = totals["true_positive"]
    tn = totals["true_negative"]
    fp = totals["false_positive"]
    fn = totals["false_negative"]
    pooled_precision = safe_divide(tp, tp + fp, empty_value=1.0)
    pooled_recall = safe_divide(tp, tp + fn, empty_value=1.0)
    pooled_specificity = safe_divide(tn, tn + fp, empty_value=1.0)
    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    pooled = {
        **totals,
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn, empty_value=1.0),
        "iou": safe_divide(tp, tp + fp + fn, empty_value=1.0),
        "precision": pooled_precision,
        "recall": pooled_recall,
        "specificity": pooled_specificity,
        "balanced_accuracy": (pooled_recall + pooled_specificity) / 2.0,
        "pixel_accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
        "mcc": safe_divide(tp * tn - fp * fn, mcc_denominator),
    }
    return {"macro_per_image": macro, "pooled_pixels": pooled}


def ground_truth_count(sample_dir: Path) -> int:
    return len(mask_paths_for(sample_dir))


def count_metrics(rows: list[dict], prediction_key: str) -> dict[str, float]:
    truth = np.asarray([row["ground_truth_count"] for row in rows], dtype=np.float64)
    prediction = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
    errors = prediction - truth
    absolute_errors = np.abs(errors)
    correlation = (
        float(np.corrcoef(truth, prediction)[0, 1])
        if truth.size > 1 and truth.std() > 0 and prediction.std() > 0
        else 0.0
    )
    return {
        "mae": float(absolute_errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(absolute_errors)),
        "mean_signed_error": float(errors.mean()),
        "mean_absolute_percentage_error": float(
            np.mean(absolute_errors / np.maximum(truth, 1.0)) * 100.0
        ),
        "within_one_count_fraction": float(np.mean(absolute_errors <= 1)),
        "within_three_count_fraction": float(np.mean(absolute_errors <= 3)),
        "pearson_correlation": correlation,
    }


def count_prediction(
    record: dict,
    minimum_size: int,
    minimum_distance: int,
    threshold_relative: float,
) -> tuple[dict, np.ndarray]:
    cleaned = remove_small_components(record["binary_prediction"], minimum_size)
    connected_count = int(label(cleaned).max())
    instances = separate_touching_nuclei(
        cleaned,
        minimum_distance=minimum_distance,
        threshold_relative=threshold_relative,
    )
    watershed_count = int(instances.max())
    truth = int(record["ground_truth_count"])
    return (
        {
            "sample_id": record["sample_id"],
            "ground_truth_count": truth,
            "connected_component_count": connected_count,
            "watershed_count": watershed_count,
            "connected_absolute_error": abs(connected_count - truth),
            "watershed_absolute_error": abs(watershed_count - truth),
            "connected_signed_error": connected_count - truth,
            "watershed_signed_error": watershed_count - truth,
        },
        instances,
    )


def evaluate_count_config(records: list[dict], settings: dict) -> tuple[list[dict], dict]:
    rows = [count_prediction(record, **settings)[0] for record in records]
    return rows, count_metrics(rows, "watershed_count")


def calibrate_counting(validation_records: list[dict]) -> tuple[dict, list[dict]]:
    trials = []
    combinations = list(
        itertools.product(
            (5, 10, 20, 30),
            (3, 4, 5, 6, 8),
            (0.05, 0.10),
        )
    )
    progress = tqdm(combinations, desc="Calibrating watershed")
    for minimum_size, minimum_distance, threshold_relative in progress:
        settings = {
            "minimum_size": minimum_size,
            "minimum_distance": minimum_distance,
            "threshold_relative": threshold_relative,
        }
        _, metrics = evaluate_count_config(validation_records, settings)
        trials.append({**settings, **metrics})
        progress.set_postfix(mae=f"{metrics['mae']:.2f}")
    trials.sort(
        key=lambda row: (
            row["mae"],
            row["rmse"],
            abs(row["mean_signed_error"]),
        )
    )
    best = trials[0]
    settings = {
        "minimum_size": int(best["minimum_size"]),
        "minimum_distance": int(best["minimum_distance"]),
        "threshold_relative": float(best["threshold_relative"]),
    }
    return settings, trials


def predict_split(
    model: torch.nn.Module,
    sample_ids: list[str],
    samples_by_id: dict[str, Path],
    mask_cache_dir: Path,
    device: torch.device,
    patch_size: int,
    overlap: int,
    tile_batch_size: int,
    threshold: float,
    predicted_mask_dir: Path | None,
    compute_segmentation: bool,
) -> tuple[list[dict], list[dict]]:
    segmentation_rows = []
    counting_records = []
    progress = tqdm(sample_ids, desc="Full-resolution inference")
    for sample_id in progress:
        sample_dir = samples_by_id[sample_id]
        image = load_rgb(image_path_for(sample_dir))
        target = np.load(
            mask_cache_dir / f"{sample_id}.npy", allow_pickle=False
        ).astype(bool)
        probability = predict_full_resolution(
            model, image, device, patch_size, overlap, tile_batch_size
        )
        binary_prediction = probability >= threshold
        if predicted_mask_dir is not None:
            Image.fromarray(binary_prediction.astype(np.uint8) * 255).save(
                predicted_mask_dir / f"{sample_id}.png"
            )

        if compute_segmentation:
            metrics = segmentation_metrics(probability, target, threshold)
            segmentation_rows.append(
                {
                    "sample_id": sample_id,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "ground_truth_foreground_pixels": int(target.sum()),
                    "predicted_foreground_pixels": int(binary_prediction.sum()),
                    **metrics,
                }
            )
        counting_records.append(
            {
                "sample_id": sample_id,
                "ground_truth_count": ground_truth_count(sample_dir),
                "binary_prediction": binary_prediction,
            }
        )
    return segmentation_rows, counting_records


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_segmentation_metrics(rows: list[dict], summary: dict, output_dir: Path) -> None:
    dice = np.asarray([row["dice"] for row in rows])
    iou = np.asarray([row["iou"] for row in rows])
    precision = np.asarray([row["precision"] for row in rows])
    recall = np.asarray([row["recall"] for row in rows])
    truth_area = np.asarray([row["ground_truth_foreground_pixels"] for row in rows])
    predicted_area = np.asarray([row["predicted_foreground_pixels"] for row in rows])
    pooled = summary["pooled_pixels"]

    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].hist(dice, bins=15, alpha=0.75, label="Dice")
    axes[0, 0].hist(iou, bins=15, alpha=0.65, label="IoU")
    axes[0, 0].set(title="Per-image overlap metrics", xlabel="Score", ylabel="Images")
    axes[0, 0].legend()

    axes[0, 1].scatter(recall, precision, alpha=0.70)
    axes[0, 1].set(
        title="Precision versus recall",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )

    upper = max(float(truth_area.max()), float(predicted_area.max()), 1.0)
    axes[1, 0].scatter(truth_area, predicted_area, alpha=0.70)
    axes[1, 0].plot([0, upper], [0, upper], "k--", linewidth=1)
    axes[1, 0].set(
        title="Segmented foreground area",
        xlabel="Annotated pixels",
        ylabel="Predicted pixels",
    )

    matrix = np.asarray(
        [
            [pooled["true_negative"], pooled["false_positive"]],
            [pooled["false_negative"], pooled["true_positive"]],
        ],
        dtype=np.float64,
    )
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    image = axes[1, 1].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(2):
        for column in range(2):
            axes[1, 1].text(
                column,
                row,
                f"{normalized[row, column]:.3f}",
                ha="center",
                va="center",
            )
    axes[1, 1].set(
        title="Row-normalized confusion matrix",
        xticks=(0, 1),
        xticklabels=("Background", "Nucleus"),
        yticks=(0, 1),
        yticklabels=("Background", "Nucleus"),
        xlabel="Predicted",
        ylabel="Annotated",
    )
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046)
    figure.tight_layout()
    figure.savefig(output_dir / "segmentation_metrics.png", dpi=180)
    plt.close(figure)


def plot_counting(rows: list[dict], output_dir: Path) -> None:
    truth = np.asarray([row["ground_truth_count"] for row in rows])
    connected = np.asarray([row["connected_component_count"] for row in rows])
    watershed = np.asarray([row["watershed_count"] for row in rows])
    upper = int(max(truth.max(), connected.max(), watershed.max(), 1))
    connected_errors = connected - truth
    watershed_errors = watershed - truth

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(truth, connected, alpha=0.65, label="Connected components")
    axes[0].scatter(truth, watershed, alpha=0.65, label="Watershed")
    axes[0].plot([0, upper], [0, upper], "k--", linewidth=1, label="Ideal")
    axes[0].set(
        title="Nucleus-count predictions",
        xlabel="Annotated count",
        ylabel="Predicted count",
    )
    axes[0].legend()

    axes[1].hist(connected_errors, bins=20, alpha=0.60, label="Connected components")
    axes[1].hist(watershed_errors, bins=20, alpha=0.60, label="Watershed")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        title="Counting errors", xlabel="Predicted minus annotated", ylabel="Images"
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "counting_metrics.png", dpi=180)
    plt.close(figure)


def save_representative_examples(
    rows: list[dict],
    samples_by_id: dict[str, Path],
    mask_cache_dir: Path,
    predicted_mask_dir: Path,
    output_dir: Path,
    number_of_examples: int,
) -> None:
    number_of_examples = min(max(number_of_examples, 1), len(rows))
    ordered = sorted(rows, key=lambda row: row["dice"])
    indices = np.linspace(0, len(ordered) - 1, number_of_examples).round().astype(int)
    chosen = [ordered[index] for index in indices]
    figure, axes = plt.subplots(
        number_of_examples, 4, figsize=(13, 3 * number_of_examples), squeeze=False
    )
    for row_index, row in enumerate(chosen):
        sample_id = row["sample_id"]
        image = load_rgb(image_path_for(samples_by_id[sample_id]))
        target = np.load(
            mask_cache_dir / f"{sample_id}.npy", allow_pickle=False
        ).astype(bool)
        with Image.open(predicted_mask_dir / f"{sample_id}.png") as predicted_image:
            prediction = np.asarray(predicted_image.convert("L")) > 0

        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(f"Image\n{sample_id[:10]}")
        axes[row_index, 1].imshow(target, cmap="gray")
        axes[row_index, 1].set_title("Ground truth")
        axes[row_index, 2].imshow(prediction, cmap="gray")
        axes[row_index, 2].set_title(f"Prediction\nDice={row['dice']:.3f}")
        axes[row_index, 3].imshow(image)
        axes[row_index, 3].imshow(prediction, cmap="autumn", alpha=0.42)
        axes[row_index, 3].set_title("Overlay")
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "representative_predictions.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for practical full-resolution testing")
    seed_everything(args.seed)
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
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("Tile overlap must be between 0 and patch-size - 1")

    model = UNet(base_channels=int(config["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split_path = args.splits or args.checkpoint.parent / "splits.json"
    with split_path.open("r", encoding="utf-8") as file:
        split_manifest = json.load(file)
    validation_ids = list(split_manifest.get("validation", []))
    test_ids = list(split_manifest.get("test", []))
    if not validation_ids or not test_ids:
        raise ValueError("Split manifest must contain validation and test IDs")
    all_split_ids = (
        list(split_manifest.get("train", [])) + validation_ids + test_ids
    )
    if len(all_split_ids) != len(set(all_split_ids)):
        raise ValueError("Split manifest contains duplicate IDs across splits")

    samples = collect_samples(args.data_dir)
    samples_by_id = {sample.name: sample for sample in samples}
    missing = [sample_id for sample_id in validation_ids + test_ids if sample_id not in samples_by_id]
    if missing:
        raise FileNotFoundError(f"Dataset is missing {len(missing)} split samples")

    mask_cache_dir = (
        args.mask_cache
        or args.checkpoint.parent / "combined_masks_original_resolution"
    )
    build_mask_cache(
        [samples_by_id[sample_id] for sample_id in validation_ids + test_ids],
        mask_cache_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predicted_mask_dir = args.output_dir / "predicted_masks"
    instance_mask_dir = args.output_dir / "instance_masks"
    predicted_mask_dir.mkdir(parents=True, exist_ok=True)
    instance_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Original-resolution test images: {len(test_ids)}")
    print("Geometric resizing: DISABLED")
    start_time = time.time()

    print("\n1/3 Predicting validation images for watershed calibration...")
    _, validation_count_records = predict_split(
        model,
        validation_ids,
        samples_by_id,
        mask_cache_dir,
        device,
        patch_size,
        overlap,
        args.tile_batch_size,
        args.threshold,
        predicted_mask_dir=None,
        compute_segmentation=False,
    )
    print("\n2/3 Selecting counting parameters on validation data only...")
    selected_parameters, parameter_trials = calibrate_counting(
        validation_count_records
    )
    validation_count_rows, validation_count_metrics = evaluate_count_config(
        validation_count_records, selected_parameters
    )

    print("\n3/3 Evaluating untouched test images...")
    segmentation_rows, test_count_records = predict_split(
        model,
        test_ids,
        samples_by_id,
        mask_cache_dir,
        device,
        patch_size,
        overlap,
        args.tile_batch_size,
        args.threshold,
        predicted_mask_dir=predicted_mask_dir,
        compute_segmentation=True,
    )
    test_count_rows = []
    for record in tqdm(test_count_records, desc="Saving instance masks"):
        row, instances = count_prediction(record, **selected_parameters)
        test_count_rows.append(row)
        Image.fromarray(instances.astype(np.uint16), mode="I;16").save(
            instance_mask_dir / f"{record['sample_id']}.tiff"
        )

    segmentation_summary = aggregate_segmentation(segmentation_rows)
    connected_summary = count_metrics(
        test_count_rows, "connected_component_count"
    )
    watershed_summary = count_metrics(test_count_rows, "watershed_count")
    elapsed_seconds = time.time() - start_time
    summary = {
        "protocol": {
            "model_selection": "Best checkpoint selected using validation Dice.",
            "semantic_threshold": args.threshold,
            "watershed_calibration": (
                "Post-processing parameters selected on validation images only."
            ),
            "final_evaluation": "Metrics reported once on untouched test images.",
            "source_images_resized": False,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "epoch": int(checkpoint["epoch"]),
            "best_validation_dice": float(checkpoint["best_val_dice"]),
        },
        "dataset": {
            "validation_samples_for_count_calibration": len(validation_ids),
            "test_samples": len(test_ids),
        },
        "inference": {
            "patch_size": patch_size,
            "tile_overlap": overlap,
            "tile_batch_size": args.tile_batch_size,
            "elapsed_seconds": elapsed_seconds,
        },
        "segmentation": segmentation_summary,
        "counting": {
            "selected_parameters": selected_parameters,
            "validation_watershed": validation_count_metrics,
            "test_connected_components": connected_summary,
            "test_watershed": watershed_summary,
        },
    }

    write_csv(args.output_dir / "per_image_segmentation.csv", segmentation_rows)
    write_csv(args.output_dir / "per_image_counting.csv", test_count_rows)
    write_csv(args.output_dir / "watershed_parameter_search.csv", parameter_trials)
    with (args.output_dir / "test_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    plot_segmentation_metrics(segmentation_rows, segmentation_summary, args.output_dir)
    plot_counting(test_count_rows, args.output_dir)
    save_representative_examples(
        segmentation_rows,
        samples_by_id,
        mask_cache_dir,
        predicted_mask_dir,
        args.output_dir,
        args.examples,
    )

    macro = segmentation_summary["macro_per_image"]
    print("\nTesting complete")
    print(f"Mean Dice: {macro['dice']['mean']:.4f}")
    print(f"Mean IoU: {macro['iou']['mean']:.4f}")
    print(f"Mean precision: {macro['precision']['mean']:.4f}")
    print(f"Mean recall: {macro['recall']['mean']:.4f}")
    print(f"Watershed count MAE: {watershed_summary['mae']:.3f}")
    print(f"Watershed median absolute error: {watershed_summary['median_absolute_error']:.3f}")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
