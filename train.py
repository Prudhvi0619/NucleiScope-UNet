"""Train a lightweight U-Net on the BBBC038 nuclei dataset.

The script expects the original Data Science Bowl directory layout:

stage1_train/
    <sample_id>/
        images/<sample_id>.png
        masks/<one PNG per nucleus>

Each set of instance masks is merged into one binary mask for semantic
segmentation. The script creates deterministic train/validation/test splits,
saves the best checkpoint, plots learning curves, and exports example test
predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net on BBBC038")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\My_Projects\Aira_matrix\stage1_train"),
        help="Path containing the 670 sample folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional small subset for a quick pipeline check.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_image(folder: Path) -> Path:
    files = sorted(
        path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(files) != 1:
        raise RuntimeError(f"Expected one image in {folder}, found {len(files)}")
    return files[0]


def collect_samples(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    samples = []
    for folder in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        image_dir = folder / "images"
        mask_dir = folder / "masks"
        if image_dir.is_dir() and mask_dir.is_dir():
            samples.append(folder)

    if not samples:
        raise RuntimeError(
            f"No valid samples found in {data_dir}. Expected images/ and masks/ "
            "inside each sample folder."
        )
    return samples


def split_samples(
    samples: list[Path], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


class NucleiDataset(Dataset):
    def __init__(
        self,
        sample_dirs: list[Path],
        image_size: int,
        augment: bool,
        mask_cache_dir: Path,
    ):
        self.sample_dirs = sample_dirs
        self.image_size = image_size
        self.augment = augment
        self.mask_cache_dir = mask_cache_dir
        self.mask_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample_dir = self.sample_dirs[index]
        image_path = find_image(sample_dir / "images")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        cached_mask_path = self.mask_cache_dir / f"{sample_dir.name}.png"
        if cached_mask_path.exists():
            combined_mask = np.asarray(
                Image.open(cached_mask_path).convert("L"), dtype=np.uint8
            ).copy()
        else:
            mask_paths = sorted(
                path
                for path in (sample_dir / "masks").iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not mask_paths:
                raise RuntimeError(f"No masks found for sample {sample_dir.name}")

            combined_mask = np.zeros((height, width), dtype=np.uint8)
            for mask_path in mask_paths:
                mask = np.asarray(
                    Image.open(mask_path).convert("L"), dtype=np.uint8
                )
                if mask.shape != combined_mask.shape:
                    raise RuntimeError(
                        f"Mask size {mask.shape} does not match image size "
                        f"{combined_mask.shape} for {sample_dir.name}"
                    )
                combined_mask = np.maximum(combined_mask, mask)
            Image.fromarray(combined_mask).save(cached_mask_path)

        if combined_mask.shape != (height, width):
            raise RuntimeError(
                f"Cached mask size {combined_mask.shape} does not match image size "
                f"{(height, width)} for {sample_dir.name}"
            )

        resize_shape = (self.image_size, self.image_size)
        image = image.resize(resize_shape, Image.Resampling.BILINEAR)
        mask_image = Image.fromarray(combined_mask).resize(
            resize_shape, Image.Resampling.NEAREST
        )

        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        mask_array = (np.asarray(mask_image, dtype=np.uint8).copy() > 0).astype(
            np.float32
        )

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        if self.augment:
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=(2,))
                mask_tensor = torch.flip(mask_tensor, dims=(2,))
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=(1,))
                mask_tensor = torch.flip(mask_tensor, dims=(1,))
            rotations = random.randint(0, 3)
            if rotations:
                image_tensor = torch.rot90(image_tensor, rotations, dims=(1, 2))
                mask_tensor = torch.rot90(mask_tensor, rotations, dims=(1, 2))

        return image_tensor.contiguous(), mask_tensor.contiguous(), sample_dir.name


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        height_difference = skip.size(2) - inputs.size(2)
        width_difference = skip.size(3) - inputs.size(3)
        inputs = F.pad(
            inputs,
            [
                width_difference // 2,
                width_difference - width_difference // 2,
                height_difference // 2,
                height_difference - height_difference // 2,
            ],
        )
        return self.conv(torch.cat((skip, inputs), dim=1))


class UNet(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.input_block = DoubleConv(3, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1 = self.input_block(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.output(x)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        dimensions = (1, 2, 3)
        intersection = (probabilities * targets).sum(dimensions)
        denominator = probabilities.sum(dimensions) + targets.sum(dimensions)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
        return bce + dice_loss.mean()


def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_images = 0
    total_intersection = 0.0
    total_prediction = 0.0
    total_target = 0.0
    total_union = 0.0

    progress = tqdm(loader, leave=False, desc="train" if training else "valid")
    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = device.type == "cuda"
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(images)
                loss = criterion(logits, masks)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = torch.sigmoid(logits) >= 0.5
        targets = masks >= 0.5
        intersection = (predictions & targets).sum().item()
        prediction_pixels = predictions.sum().item()
        target_pixels = targets.sum().item()
        union = (predictions | targets).sum().item()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size
        total_intersection += intersection
        total_prediction += prediction_pixels
        total_target += target_pixels
        total_union += union
        progress.set_postfix(loss=f"{loss.item():.4f}")

    epsilon = 1e-7
    dice = (2.0 * total_intersection + epsilon) / (
        total_prediction + total_target + epsilon
    )
    iou = (total_intersection + epsilon) / (total_union + epsilon)
    return {
        "loss": total_loss / max(total_images, 1),
        "dice": dice,
        "iou": iou,
    }


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["val_dice"] for row in history], label="Dice")
    axes[1].plot(epochs, [row["val_iou"] for row in history], label="IoU")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)


@torch.inference_mode()
def save_prediction_examples(
    model: nn.Module,
    dataset: NucleiDataset,
    device: torch.device,
    output_dir: Path,
    number_of_examples: int = 4,
) -> None:
    model.eval()
    count = min(number_of_examples, len(dataset))
    figure, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)

    for row in range(count):
        image, mask, sample_id = dataset[row]
        logits = model(image.unsqueeze(0).to(device))
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prediction = probability >= 0.5

        axes[row, 0].imshow(image.permute(1, 2, 0).numpy())
        axes[row, 0].set_title(f"Image\n{sample_id[:10]}")
        axes[row, 1].imshow(mask[0].numpy(), cmap="gray")
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(probability, cmap="viridis", vmin=0, vmax=1)
        axes[row, 2].set_title("Probability")
        axes[row, 3].imshow(image.permute(1, 2, 0).numpy())
        axes[row, 3].imshow(prediction, cmap="autumn", alpha=0.45)
        axes[row, 3].set_title("Prediction overlay")
        for axis in axes[row]:
            axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_dir / "test_predictions.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.data_dir)
    if args.max_samples is not None:
        if args.max_samples < 10:
            raise ValueError("--max-samples must be at least 10")
        samples = samples[: args.max_samples]

    train_samples, val_samples, test_samples = split_samples(samples, args.seed)
    print(f"Dataset: {args.data_dir}")
    print(
        f"Samples: train={len(train_samples)}, validation={len(val_samples)}, "
        f"test={len(test_samples)}"
    )

    split_manifest = {
        "train": [sample.name for sample in train_samples],
        "validation": [sample.name for sample in val_samples],
        "test": [sample.name for sample in test_samples],
    }
    with (args.output_dir / "splits.json").open("w", encoding="utf-8") as file:
        json.dump(split_manifest, file, indent=2)

    mask_cache_dir = args.output_dir / "combined_masks"
    train_dataset = NucleiDataset(
        train_samples, args.image_size, augment=True, mask_cache_dir=mask_cache_dir
    )
    val_dataset = NucleiDataset(
        val_samples, args.image_size, augment=False, mask_cache_dir=mask_cache_dir
    )
    test_dataset = NucleiDataset(
        test_samples, args.image_size, augment=False, mask_cache_dir=mask_cache_dir
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and "
            "confirm torch.cuda.is_available() returns True."
        )

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Mixed precision: enabled")
    torch.backends.cudnn.benchmark = True

    model = UNet(base_channels=32).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    scaler = create_grad_scaler(enabled=True)

    best_dice = -1.0
    history: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["dice"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {row['train_loss']:.4f}, dice {row['train_dice']:.4f} | "
            f"val loss {row['val_loss']:.4f}, dice {row['val_dice']:.4f}, "
            f"IoU {row['val_iou']:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_dice": val_metrics["dice"],
            "image_size": args.image_size,
            "base_channels": 32,
        }
        torch.save(checkpoint, args.output_dir / "last_model.pt")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(checkpoint, args.output_dir / "best_model.pt")
            print(f"  Saved new best checkpoint (Dice={best_dice:.4f})")

        save_history(history, args.output_dir)

    best_checkpoint = torch.load(
        args.output_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device)
    elapsed_minutes = (time.time() - start_time) / 60.0

    final_results = {
        "best_validation_dice": best_dice,
        "test_loss": test_metrics["loss"],
        "test_dice": test_metrics["dice"],
        "test_iou": test_metrics["iou"],
        "training_minutes": elapsed_minutes,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "device": torch.cuda.get_device_name(0),
    }
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=2)

    save_prediction_examples(model, test_dataset, device, args.output_dir)
    print("\nTraining complete")
    print(json.dumps(final_results, indent=2))
    print(f"Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
