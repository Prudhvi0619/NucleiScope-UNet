"""Evaluate semantic segmentation, instance segmentation, and nucleus counting."""

from __future__ import annotations

import argparse
import csv
import itertools
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
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.measure import label
from tqdm import tqdm

from nuclei_io import (
    atomic_json_dump,
    build_dataset_manifest,
    build_mask_cache,
    collect_samples,
    ensure_new_output_directory,
    image_path_for,
    load_cached_centers,
    load_cached_mask,
    load_instance_labels,
    load_rgb,
    save_instance_labels,
    sha256_file,
)
from postprocess import remove_small_components, separate_touching_nuclei
from train import (
    UNet,
    configure_reproducibility,
    load_checkpoint,
    predict_full_resolution,
    resolve_device,
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
INSTANCE_THRESHOLDS = tuple(float(value) for value in np.arange(0.50, 1.00, 0.05))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-calibrated held-out evaluation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage1_train"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/run/best_model.pt"))
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--mask-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Predeclare a threshold; otherwise select it on validation data.",
    )
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(empty_value)


def segmentation_metrics(probability, target, threshold) -> dict[str, float | int]:
    target = np.asarray(target, dtype=bool)
    prediction = probability >= threshold
    tp = int(np.logical_and(prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    recall = safe_divide(tp, tp + fn, 1.0)
    specificity = safe_divide(tn, tn + fp, 1.0)
    clipped = np.clip(probability.astype(np.float64), 1e-7, 1 - 1e-7)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    target_area = int(target.sum())
    area_error = (
        safe_divide(abs(int(prediction.sum()) - target_area) * 100.0, target_area)
        if target_area
        else (0.0 if not prediction.any() else float("inf"))
    )
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn, 1.0),
        "iou": safe_divide(tp, tp + fp + fn, 1.0),
        "precision": safe_divide(tp, tp + fp, 1.0),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "pixel_accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
        "mcc": safe_divide(tp * tn - fp * fn, mcc_denominator),
        "binary_cross_entropy": float(
            -np.mean(target * np.log(clipped) + (~target) * np.log(1 - clipped))
        ),
        "foreground_area_error_percent": area_error,
    }


def aggregate_segmentation(rows: list[dict]) -> dict:
    macro = {}
    for metric in SEGMENTATION_METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        macro[metric] = {
            "mean": float(finite.mean()),
            "standard_deviation": float(finite.std()),
            "median": float(np.median(finite)),
            "minimum": float(finite.min()),
            "maximum": float(finite.max()),
            "finite_samples": int(len(finite)),
        }
    totals = {
        key: int(sum(row[key] for row in rows))
        for key in ("true_positive", "true_negative", "false_positive", "false_negative")
    }
    tp, tn, fp, fn = (
        totals["true_positive"],
        totals["true_negative"],
        totals["false_positive"],
        totals["false_negative"],
    )
    precision, recall = safe_divide(tp, tp + fp, 1.0), safe_divide(tp, tp + fn, 1.0)
    specificity = safe_divide(tn, tn + fp, 1.0)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    pooled = {
        **totals,
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn, 1.0),
        "iou": safe_divide(tp, tp + fp + fn, 1.0),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "pixel_accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
        "mcc": safe_divide(tp * tn - fp * fn, denominator),
    }
    return {"macro_per_image": macro, "pooled_pixels": pooled}


def instance_metrics(
    truth_labels: np.ndarray, predicted_labels: np.ndarray
) -> dict[str, float | int]:
    truth_count = int(truth_labels.max(initial=0))
    predicted_count = int(predicted_labels.max(initial=0))
    if truth_count and predicted_count:
        encoded = (
            truth_labels.astype(np.int64).ravel() * (predicted_count + 1) + predicted_labels.ravel()
        )
        intersections = np.bincount(
            encoded, minlength=(truth_count + 1) * (predicted_count + 1)
        ).reshape(truth_count + 1, predicted_count + 1)[1:, 1:]
        truth_areas = np.bincount(truth_labels.ravel(), minlength=truth_count + 1)[1:, None]
        predicted_areas = np.bincount(predicted_labels.ravel(), minlength=predicted_count + 1)[
            None, 1:
        ]
        iou = intersections / np.maximum(truth_areas + predicted_areas - intersections, 1)
        rows, columns = linear_sum_assignment(-iou)
        assigned = iou[rows, columns]
    else:
        assigned = np.asarray([], dtype=np.float64)
    output: dict[str, float | int] = {
        "ground_truth_instances": truth_count,
        "predicted_instances": predicted_count,
    }
    average_precisions = []
    for threshold in INSTANCE_THRESHOLDS:
        tp = int(np.sum(assigned >= threshold))
        fp, fn = predicted_count - tp, truth_count - tp
        score = safe_divide(tp, tp + fp + fn, 1.0 if truth_count == predicted_count == 0 else 0.0)
        output[f"instance_ap_{threshold:.2f}"] = score
        average_precisions.append(score)
        if math.isclose(threshold, 0.5):
            output.update(
                {
                    "instance_true_positive_0.50": tp,
                    "instance_false_positive_0.50": fp,
                    "instance_false_negative_0.50": fn,
                    "instance_precision_0.50": safe_divide(tp, tp + fp, 1.0),
                    "instance_recall_0.50": safe_divide(tp, tp + fn, 1.0),
                    "instance_f1_0.50": safe_divide(2 * tp, 2 * tp + fp + fn, 1.0),
                    "mean_matched_iou_0.50": float(assigned[assigned >= 0.5].mean())
                    if np.any(assigned >= 0.5)
                    else 0.0,
                }
            )
    output["instance_mean_ap_0.50_0.95"] = float(np.mean(average_precisions))
    return output


def aggregate_instance(rows: list[dict]) -> dict:
    fields = (
        "instance_precision_0.50",
        "instance_recall_0.50",
        "instance_f1_0.50",
        "mean_matched_iou_0.50",
        "instance_mean_ap_0.50_0.95",
    )
    macro = {field: float(np.mean([row[field] for row in rows])) for field in fields}
    tp = sum(row["instance_true_positive_0.50"] for row in rows)
    fp = sum(row["instance_false_positive_0.50"] for row in rows)
    fn = sum(row["instance_false_negative_0.50"] for row in rows)
    return {
        "macro_per_image": macro,
        "pooled_at_iou_0.50": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": safe_divide(tp, tp + fp, 1.0),
            "recall": safe_divide(tp, tp + fn, 1.0),
            "f1": safe_divide(2 * tp, 2 * tp + fp + fn, 1.0),
        },
    }


def count_metrics(rows: list[dict], prediction_key: str) -> dict[str, float]:
    truth = np.asarray([row["ground_truth_count"] for row in rows], np.float64)
    prediction = np.asarray([row[prediction_key] for row in rows], np.float64)
    errors, absolute = prediction - truth, np.abs(prediction - truth)
    correlation = (
        float(np.corrcoef(truth, prediction)[0, 1]) if truth.std() and prediction.std() else 0.0
    )
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(absolute)),
        "mean_signed_error": float(errors.mean()),
        "mean_absolute_percentage_error": float(np.mean(absolute / np.maximum(truth, 1.0)) * 100),
        "within_one_count_fraction": float(np.mean(absolute <= 1)),
        "within_three_count_fraction": float(np.mean(absolute <= 3)),
        "exact_count_fraction": float(np.mean(absolute == 0)),
        "pearson_correlation": correlation,
    }


def count_prediction(record: dict, settings: dict) -> tuple[dict, np.ndarray]:
    connected_cleaned = remove_small_components(
        record["binary_prediction"], settings["connected_minimum_size"]
    )
    connected_count = int(label(connected_cleaned).max())
    watershed_cleaned = remove_small_components(
        record["binary_prediction"], settings["watershed_minimum_size"]
    )
    instances = separate_touching_nuclei(
        watershed_cleaned, settings["minimum_distance"], settings["threshold_relative"]
    )
    watershed_count = int(instances.max(initial=0))
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


def calibrate_threshold(records: list[dict]) -> tuple[float, list[dict]]:
    trials = []
    for threshold in np.arange(0.30, 0.701, 0.05):
        dice = [
            segmentation_metrics(record["probability"], record["target"], threshold)["dice"]
            for record in records
        ]
        trials.append({"threshold": float(round(threshold, 2)), "macro_dice": float(np.mean(dice))})
    trials.sort(key=lambda row: (-row["macro_dice"], abs(row["threshold"] - 0.5)))
    return float(trials[0]["threshold"]), trials


def calibrate_counting(records: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    connected_trials = []
    for minimum_size in (1, 5, 10, 20, 30):
        rows = []
        settings = {
            "connected_minimum_size": minimum_size,
            "watershed_minimum_size": 5,
            "minimum_distance": 8,
            "threshold_relative": 0.05,
        }
        for record in records:
            rows.append(count_prediction(record, settings)[0])
        connected_trials.append(
            {"minimum_size": minimum_size, **count_metrics(rows, "connected_component_count")}
        )
    connected_trials.sort(key=lambda row: (row["mae"], row["rmse"], abs(row["mean_signed_error"])))
    connected_minimum_size = int(connected_trials[0]["minimum_size"])

    watershed_trials = []
    for minimum_size, minimum_distance, threshold_relative in itertools.product(
        (1, 5, 10, 20, 30), (3, 4, 5, 6, 8, 10), (0.05, 0.10, 0.20)
    ):
        settings = {
            "connected_minimum_size": connected_minimum_size,
            "watershed_minimum_size": minimum_size,
            "minimum_distance": minimum_distance,
            "threshold_relative": threshold_relative,
        }
        rows = [count_prediction(record, settings)[0] for record in records]
        watershed_trials.append(
            {
                "minimum_size": minimum_size,
                "minimum_distance": minimum_distance,
                "threshold_relative": threshold_relative,
                **count_metrics(rows, "watershed_count"),
            }
        )
    watershed_trials.sort(key=lambda row: (row["mae"], row["rmse"], abs(row["mean_signed_error"])))
    best = watershed_trials[0]
    return (
        {
            "connected_minimum_size": connected_minimum_size,
            "watershed_minimum_size": int(best["minimum_size"]),
            "minimum_distance": int(best["minimum_distance"]),
            "threshold_relative": float(best["threshold_relative"]),
        },
        connected_trials,
        watershed_trials,
    )


@torch.inference_mode()
def predict_records(
    model, sample_ids, samples_by_id, cache_dir, device, patch_size, overlap, tile_batch_size
):
    records = []
    for sample_id in tqdm(sample_ids, desc="Full-resolution inference"):
        sample = samples_by_id[sample_id]
        image = load_rgb(image_path_for(sample))
        target = load_cached_mask(cache_dir, sample_id).astype(bool)
        probability = predict_full_resolution(
            model, image, device, patch_size, overlap, tile_batch_size
        )
        records.append(
            {
                "sample_id": sample_id,
                "image": image,
                "target": target,
                "probability": probability,
                "ground_truth_count": int(len(load_cached_centers(cache_dir, sample_id))),
            }
        )
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = generator.choice(values, size=len(values), replace=True).mean()
    low, high = np.percentile(estimates, (2.5, 97.5))
    return {
        "estimate": float(values.mean()),
        "confidence_level": 0.95,
        "lower": float(low),
        "upper": float(high),
        "bootstrap_samples": samples,
    }


def save_representative_examples(rows, records_by_id, output_dir, number_of_examples):
    ordered = sorted(rows, key=lambda row: row["dice"])
    number_of_examples = min(max(number_of_examples, 1), len(ordered))
    indices = np.linspace(0, len(ordered) - 1, number_of_examples).round().astype(int)
    figure, axes = plt.subplots(
        number_of_examples, 4, figsize=(13, 3 * number_of_examples), squeeze=False
    )
    for row_index, index in enumerate(indices):
        row = ordered[index]
        record = records_by_id[row["sample_id"]]
        image, target = record["image"], record["target"]
        prediction = record["binary_prediction"]
        overlay = image.astype(np.float32).copy()
        overlay[prediction] = 0.58 * overlay[prediction] + 0.42 * np.asarray([255, 65, 20])
        panels = (image, target, prediction, overlay.astype(np.uint8))
        titles = (
            f"Image\n{row['sample_id'][:10]}",
            "Ground truth",
            f"Prediction\nDice={row['dice']:.3f}",
            "Overlay",
        )
        for column, (panel, title) in enumerate(zip(panels, titles, strict=False)):
            axes[row_index, column].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axes[row_index, column].set_title(title)
            axes[row_index, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "representative_predictions.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.threshold is not None and not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    if args.tile_batch_size < 1 or args.examples < 1 or args.bootstrap_samples < 100:
        raise ValueError("Tile batch, examples, and bootstrap sample counts must be positive")
    device = resolve_device(args.device)
    configure_reproducibility(args.seed, True)
    checkpoint = load_checkpoint(args.checkpoint, device)
    contract = checkpoint["run_contract"]
    patch_size = int(contract["patch_size"])
    overlap = (
        int(args.tile_overlap) if args.tile_overlap is not None else int(contract["tile_overlap"])
    )
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("Invalid tile overlap")

    samples = collect_samples(args.data_dir)
    if contract.get("max_samples") is not None:
        samples = samples[: int(contract["max_samples"])]
    dataset_manifest = build_dataset_manifest(samples)
    if dataset_manifest["dataset_fingerprint"] != checkpoint["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint does not match the checkpoint")
    split_manifest = checkpoint["split_manifest"]
    if args.splits is not None:
        with args.splits.open("r", encoding="utf-8") as file:
            external_split = json.load(file)
        if external_split.get("split_fingerprint") != split_manifest.get("split_fingerprint"):
            raise ValueError("External split manifest does not match the checkpoint")
        split_manifest = external_split
    all_ids = split_manifest["train"] + split_manifest["validation"] + split_manifest["test"]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != {sample.name for sample in samples}:
        raise ValueError("Split manifest is incomplete, duplicated, or belongs to another dataset")

    ensure_new_output_directory(args.output_dir)
    cache_dir = args.mask_cache or args.output_dir / "validated_mask_cache"
    build_mask_cache(samples, cache_dir, dataset_manifest)
    samples_by_id = {sample.name: sample for sample in samples}
    model = UNet(int(contract["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    start = time.time()
    validation_records = predict_records(
        model,
        split_manifest["validation"],
        samples_by_id,
        cache_dir,
        device,
        patch_size,
        overlap,
        args.tile_batch_size,
    )
    if args.threshold is None:
        threshold, threshold_trials = calibrate_threshold(validation_records)
        threshold_source = "selected by validation macro Dice"
    else:
        threshold, threshold_trials = float(args.threshold), []
        threshold_source = "predeclared command-line value"
    for record in validation_records:
        record["binary_prediction"] = record["probability"] >= threshold
    settings, connected_trials, watershed_trials = calibrate_counting(validation_records)
    validation_count_rows = [count_prediction(record, settings)[0] for record in validation_records]

    test_records = predict_records(
        model,
        split_manifest["test"],
        samples_by_id,
        cache_dir,
        device,
        patch_size,
        overlap,
        args.tile_batch_size,
    )
    predicted_dir, instance_dir = (
        args.output_dir / "predicted_masks",
        args.output_dir / "instance_masks",
    )
    predicted_dir.mkdir()
    instance_dir.mkdir()
    segmentation_rows, counting_rows, instance_rows = [], [], []
    records_by_id = {}
    for record in tqdm(test_records, desc="Scoring and saving test predictions"):
        record["binary_prediction"] = record["probability"] >= threshold
        sample_id = record["sample_id"]
        metrics = segmentation_metrics(record["probability"], record["target"], threshold)
        segmentation_rows.append(
            {
                "sample_id": sample_id,
                "width": record["image"].shape[1],
                "height": record["image"].shape[0],
                "ground_truth_foreground_pixels": int(record["target"].sum()),
                "predicted_foreground_pixels": int(record["binary_prediction"].sum()),
                **metrics,
            }
        )
        count_row, predicted_instances = count_prediction(record, settings)
        object_metrics = instance_metrics(
            load_instance_labels(samples_by_id[sample_id]), predicted_instances
        )
        count_row.update(object_metrics)
        counting_rows.append(count_row)
        instance_rows.append({"sample_id": sample_id, **object_metrics})
        Image.fromarray(record["binary_prediction"].astype(np.uint8) * 255).save(
            predicted_dir / f"{sample_id}.png"
        )
        save_instance_labels(predicted_instances, instance_dir / f"{sample_id}.tiff")
        records_by_id[sample_id] = record

    segmentation_summary = aggregate_segmentation(segmentation_rows)
    instance_summary = aggregate_instance(instance_rows)
    connected_summary = count_metrics(counting_rows, "connected_component_count")
    watershed_summary = count_metrics(counting_rows, "watershed_count")
    checkpoint_sha = sha256_file(args.checkpoint)
    calibration = {
        "schema_version": 2,
        "checkpoint_sha256": checkpoint_sha,
        "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
        "split_fingerprint": split_manifest["split_fingerprint"],
        "semantic_threshold": threshold,
        "semantic_threshold_source": threshold_source,
        "counting": settings,
        "pixel_scale_warning": "Counting settings are in pixels and must be recalibrated when physical scale changes.",
    }
    atomic_json_dump(calibration, args.output_dir / "calibration.json")
    summary = {
        "schema_version": 2,
        "protocol": {
            "model_selection": "Best checkpoint selected using validation macro Dice.",
            "semantic_threshold": threshold_source,
            "postprocessing": "Connected-components and watershed settings calibrated independently on validation data.",
            "final_evaluation": "Held-out test predictions scored once after calibration.",
        },
        "provenance": {
            "checkpoint_name": args.checkpoint.name,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
            "split_fingerprint": split_manifest["split_fingerprint"],
            "training_source_fingerprint": checkpoint["source_fingerprint"],
        },
        "dataset": {
            "validation_samples": len(validation_records),
            "test_samples": len(test_records),
        },
        "inference": {
            "device": str(device),
            "patch_size": patch_size,
            "tile_overlap": overlap,
            "tile_batch_size": args.tile_batch_size,
            "elapsed_seconds": time.time() - start,
        },
        "segmentation": segmentation_summary,
        "instance_segmentation": instance_summary,
        "counting": {
            "selected_parameters": settings,
            "validation_connected_components": count_metrics(
                validation_count_rows, "connected_component_count"
            ),
            "validation_watershed": count_metrics(validation_count_rows, "watershed_count"),
            "test_connected_components": connected_summary,
            "test_watershed": watershed_summary,
        },
        "confidence_intervals": {
            "mean_dice": bootstrap_ci(
                [row["dice"] for row in segmentation_rows], args.bootstrap_samples, args.seed
            ),
            "mean_instance_ap": bootstrap_ci(
                [row["instance_mean_ap_0.50_0.95"] for row in instance_rows],
                args.bootstrap_samples,
                args.seed + 1,
            ),
            "watershed_mae": bootstrap_ci(
                [row["watershed_absolute_error"] for row in counting_rows],
                args.bootstrap_samples,
                args.seed + 2,
            ),
        },
    }
    write_csv(args.output_dir / "per_image_segmentation.csv", segmentation_rows)
    write_csv(args.output_dir / "per_image_counting_and_instances.csv", counting_rows)
    write_csv(args.output_dir / "validation_counting.csv", validation_count_rows)
    write_csv(args.output_dir / "semantic_threshold_search.csv", threshold_trials)
    write_csv(args.output_dir / "connected_component_search.csv", connected_trials)
    write_csv(args.output_dir / "watershed_parameter_search.csv", watershed_trials)
    atomic_json_dump(summary, args.output_dir / "test_summary.json")
    save_representative_examples(segmentation_rows, records_by_id, args.output_dir, args.examples)
    print(
        json.dumps(
            {
                "mean_dice": segmentation_summary["macro_per_image"]["dice"]["mean"],
                "mean_instance_ap": instance_summary["macro_per_image"][
                    "instance_mean_ap_0.50_0.95"
                ],
                "watershed_count_mae": watershed_summary["mae"],
                "calibration": str((args.output_dir / "calibration.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
