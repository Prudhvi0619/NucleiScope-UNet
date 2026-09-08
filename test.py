"""Evaluate the best NucleiScope checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from train import BCEDiceLoss, NucleiDataset, UNet, collect_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained NucleiScope model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\My_Projects\Aira_matrix\stage1_train"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/best_model.pt")
    )
    parser.add_argument("--splits", type=Path, default=Path("outputs/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=8)
    return parser.parse_args()


def binary_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    true_positive = np.logical_and(prediction, target).sum(dtype=np.float64)
    false_positive = np.logical_and(prediction, ~target).sum(dtype=np.float64)
    false_negative = np.logical_and(~prediction, target).sum(dtype=np.float64)
    true_negative = np.logical_and(~prediction, ~target).sum(dtype=np.float64)
    epsilon = 1e-7

    return {
        "dice": float(
            (2.0 * true_positive + epsilon)
            / (2.0 * true_positive + false_positive + false_negative + epsilon)
        ),
        "iou": float(
            (true_positive + epsilon)
            / (true_positive + false_positive + false_negative + epsilon)
        ),
        "precision": float(
            (true_positive + epsilon) / (true_positive + false_positive + epsilon)
        ),
        "recall": float(
            (true_positive + epsilon) / (true_positive + false_negative + epsilon)
        ),
        "pixel_accuracy": float(
            (true_positive + true_negative + epsilon)
            / (
                true_positive
                + true_negative
                + false_positive
                + false_negative
                + epsilon
            )
        ),
    }


def save_examples(
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    threshold: float,
    output_path: Path,
) -> None:
    if not examples:
        return

    figure, axes = plt.subplots(
        len(examples), 4, figsize=(12, 3 * len(examples)), squeeze=False
    )
    for row, (image, target, probability, sample_id) in enumerate(examples):
        prediction = probability >= threshold
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Image\n{sample_id[:10]}")
        axes[row, 1].imshow(target, cmap="gray")
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(probability, cmap="viridis", vmin=0, vmax=1)
        axes[row, 2].set_title("Probability")
        axes[row, 3].imshow(image)
        axes[row, 3].imshow(prediction, cmap="autumn", alpha=0.45)
        axes[row, 3].set_title("Prediction overlay")
        for axis in axes[row]:
            axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.splits.is_file():
        raise FileNotFoundError(f"Split manifest not found: {args.splits}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 256))
    base_channels = int(checkpoint.get("base_channels", 32))

    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with args.splits.open("r", encoding="utf-8") as file:
        split_manifest = json.load(file)
    test_ids = split_manifest.get("test", [])
    if not test_ids:
        raise RuntimeError("The split manifest contains no test sample IDs")

    available = {sample.name: sample for sample in collect_samples(args.data_dir)}
    missing = [sample_id for sample_id in test_ids if sample_id not in available]
    if missing:
        raise RuntimeError(f"Test samples missing from dataset: {missing[:5]}")
    test_samples = [available[sample_id] for sample_id in test_ids]

    mask_cache_dir = args.checkpoint.parent / "combined_masks"
    dataset = NucleiDataset(
        test_samples,
        image_size=image_size,
        augment=False,
        mask_cache_dir=mask_cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir = args.output_dir / "predicted_masks"
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    criterion = BCEDiceLoss()
    rows: list[dict[str, float | str]] = []
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    total_loss = 0.0
    total_images = 0

    print(f"Device: {device}")
    print(f"Testing {len(dataset)} held-out samples")
    for images, masks, sample_ids in tqdm(loader, desc="test"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        probabilities = torch.sigmoid(logits).cpu().numpy()
        target_arrays = masks.cpu().numpy()
        image_arrays = images.cpu().permute(0, 2, 3, 1).numpy()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

        for index, sample_id in enumerate(sample_ids):
            probability = probabilities[index, 0]
            prediction = probability >= args.threshold
            target = target_arrays[index, 0] >= 0.5
            metrics = binary_metrics(prediction, target)
            rows.append(
                {
                    "sample_id": sample_id,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "pixel_accuracy": metrics["pixel_accuracy"],
                }
            )
            Image.fromarray(prediction.astype(np.uint8) * 255).save(
                mask_output_dir / f"{sample_id}.png"
            )

            if len(examples) < args.examples:
                examples.append(
                    (image_arrays[index], target, probability, str(sample_id))
                )

    metric_names = ["dice", "iou", "precision", "recall", "pixel_accuracy"]
    summary: dict[str, float | int | str] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "test_samples": len(rows),
        "threshold": args.threshold,
        "test_loss": total_loss / max(total_images, 1),
    }
    for metric_name in metric_names:
        values = np.asarray([float(row[metric_name]) for row in rows])
        summary[f"mean_{metric_name}"] = float(values.mean())
        summary[f"std_{metric_name}"] = float(values.std())

    with (args.output_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "test_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_examples(
        examples, args.threshold, args.output_dir / "test_examples.png"
    )
    print(json.dumps(summary, indent=2))
    print(f"Testing outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
