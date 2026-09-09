"""Train U-Net on original-resolution microscopy images using random patches.

No image or mask is geometrically resized. Training uses pixel-preserving random
crops, while validation uses overlapping tiles reconstructed at the original
image resolution. The held-out test split is recorded but never used here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
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
    parser = argparse.ArgumentParser(
        description="Patch-wise U-Net training without resizing source images"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/stage1_train"),
        help="Directory containing the 670 annotated sample folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument(
        "--patches-per-image",
        type=int,
        default=2,
        help="Random training patches sampled per source image in each epoch.",
    )
    parser.add_argument(
        "--positive-crop-probability",
        type=float,
        default=0.75,
        help="Probability of centering a training crop near a nucleus pixel.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=64,
        help="Overlap used for full-resolution validation inference.",
    )
    parser.add_argument(
        "--validation-tile-batch",
        type=int,
        default=4,
        help="Number of validation tiles inferred together.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=8,
        help="Stop after this many epochs without improvement; use 0 to disable.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a last_model.pt checkpoint.",
    )
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
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def image_path_for(sample_dir: Path) -> Path:
    image_dir = sample_dir / "images"
    candidates = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one image in {image_dir}, found {len(candidates)}"
        )
    return candidates[0]


def mask_paths_for(sample_dir: Path) -> list[Path]:
    mask_dir = sample_dir / "masks"
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
        raise ValueError(f"Expected at least 10 samples, found {len(samples)}")
    return samples


def split_samples(
    samples: Sequence[Path], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    train_end = int(0.80 * len(shuffled))
    validation_end = int(0.90 * len(shuffled))
    return (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )


def load_rgb(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def build_mask_cache(samples: Sequence[Path], cache_dir: Path) -> None:
    """Merge instance masks once and cache them at their original resolution."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(samples, desc="Preparing original-resolution masks")
    for sample_dir in progress:
        destination = cache_dir / f"{sample_dir.name}.npy"
        if destination.exists():
            cached = np.load(destination, mmap_mode="r")
            with Image.open(image_path_for(sample_dir)) as image:
                expected_shape = (image.height, image.width)
            if cached.shape == expected_shape:
                continue
            destination.unlink()

        with Image.open(image_path_for(sample_dir)) as image:
            expected_shape = (image.height, image.width)
        combined = np.zeros(expected_shape, dtype=np.uint8)
        for mask_path in mask_paths_for(sample_dir):
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
            if mask.shape != expected_shape:
                raise ValueError(
                    f"Mask {mask_path} has shape {mask.shape}; expected {expected_shape}"
                )
            combined[mask > 0] = 1
        np.save(destination, combined, allow_pickle=False)


def pad_to_patch(
    image: np.ndarray, mask: np.ndarray, patch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    pad_bottom = max(0, patch_size - height)
    pad_right = max(0, patch_size - width)
    if pad_bottom == 0 and pad_right == 0:
        return image, mask

    # Reflection preserves image texture. The ground-truth mask is zero-padded so
    # padding never invents nuclei.
    image_mode = "reflect" if height > 1 and width > 1 else "edge"
    image = np.pad(
        image,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode=image_mode,
    )
    mask = np.pad(mask, ((0, pad_bottom), (0, pad_right)), mode="constant")
    return image, mask


class RandomPatchDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Path],
        mask_cache_dir: Path,
        patch_size: int,
        patches_per_image: int,
        positive_crop_probability: float,
        augment: bool,
    ) -> None:
        self.samples = list(samples)
        self.mask_cache_dir = mask_cache_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.positive_crop_probability = positive_crop_probability
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def _crop_origin(self, mask: np.ndarray) -> tuple[int, int]:
        patch = self.patch_size
        height, width = mask.shape
        if random.random() < self.positive_crop_probability and mask.any():
            ys, xs = np.nonzero(mask)
            selected = random.randrange(len(ys))
            jitter = patch // 4
            center_y = int(ys[selected]) + random.randint(-jitter, jitter)
            center_x = int(xs[selected]) + random.randint(-jitter, jitter)
            top = max(0, min(center_y - patch // 2, height - patch))
            left = max(0, min(center_x - patch // 2, width - patch))
        else:
            top = random.randint(0, max(0, height - patch))
            left = random.randint(0, max(0, width - patch))
        return top, left

    def __getitem__(self, index: int):
        sample_dir = self.samples[index % len(self.samples)]
        image = load_rgb(image_path_for(sample_dir))
        mask = np.load(
            self.mask_cache_dir / f"{sample_dir.name}.npy", allow_pickle=False
        )
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask size mismatch for {sample_dir.name}")

        image, mask = pad_to_patch(image, mask, self.patch_size)
        top, left = self._crop_origin(mask)
        bottom = top + self.patch_size
        right = left + self.patch_size
        image = image[top:bottom, left:right]
        mask = mask[top:bottom, left:right]

        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(
            2, 0, 1
        )
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

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

            # Mild intensity augmentation; spatial scale is never changed.
            if random.random() < 0.4:
                contrast = random.uniform(0.90, 1.10)
                brightness = random.uniform(-0.05, 0.05)
                image_tensor = torch.clamp(image_tensor * contrast + brightness, 0, 1)

        return image_tensor.contiguous(), mask_tensor.contiguous()


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
        difference_y = skip.size(2) - inputs.size(2)
        difference_x = skip.size(3) - inputs.size(3)
        inputs = F.pad(
            inputs,
            [
                difference_x // 2,
                difference_x - difference_x // 2,
                difference_y // 2,
                difference_y - difference_y // 2,
            ],
        )
        return self.conv(torch.cat([skip, inputs], dim=1))


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


def training_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    intersection = prediction_pixels = target_pixels = union = 0.0

    progress = tqdm(loader, desc="Training patches", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = torch.sigmoid(logits.detach()) >= 0.5
        targets = masks >= 0.5
        intersection += (predictions & targets).sum().item()
        prediction_pixels += predictions.sum().item()
        target_pixels += targets.sum().item()
        union += (predictions | targets).sum().item()
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    epsilon = 1e-7
    return {
        "loss": total_loss / max(total_items, 1),
        "dice": (2 * intersection + epsilon)
        / (prediction_pixels + target_pixels + epsilon),
        "iou": (intersection + epsilon) / (union + epsilon),
    }


def tile_origins(length: int, patch_size: int, overlap: int) -> list[int]:
    if length <= patch_size:
        return [0]
    stride = patch_size - overlap
    origins = list(range(0, length - patch_size + 1, stride))
    final_origin = length - patch_size
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


@torch.inference_mode()
def predict_full_resolution(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    overlap: int,
    tile_batch_size: int,
) -> np.ndarray:
    """Stitch overlapping predictions without resizing the source image."""
    original_height, original_width = image.shape[:2]
    dummy_mask = np.zeros((original_height, original_width), dtype=np.uint8)
    padded_image, _ = pad_to_patch(image, dummy_mask, patch_size)
    height, width = padded_image.shape[:2]
    y_origins = tile_origins(height, patch_size, overlap)
    x_origins = tile_origins(width, patch_size, overlap)

    probability_sum = np.zeros((height, width), dtype=np.float32)
    contribution_count = np.zeros((height, width), dtype=np.float32)
    tiles: list[np.ndarray] = []
    locations: list[tuple[int, int]] = []

    def infer_pending() -> None:
        if not tiles:
            return
        batch = np.stack(tiles).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            probabilities = torch.sigmoid(model(tensor))[:, 0]
        probabilities_np = probabilities.float().cpu().numpy()
        for probability, (top, left) in zip(probabilities_np, locations):
            probability_sum[
                top : top + patch_size, left : left + patch_size
            ] += probability
            contribution_count[
                top : top + patch_size, left : left + patch_size
            ] += 1.0
        tiles.clear()
        locations.clear()

    for top in y_origins:
        for left in x_origins:
            tiles.append(
                padded_image[
                    top : top + patch_size, left : left + patch_size
                ]
            )
            locations.append((top, left))
            if len(tiles) == tile_batch_size:
                infer_pending()
    infer_pending()

    probability = probability_sum / np.maximum(contribution_count, 1.0)
    return probability[:original_height, :original_width]


@torch.inference_mode()
def validate_full_images(
    model: nn.Module,
    samples: Sequence[Path],
    mask_cache_dir: Path,
    device: torch.device,
    patch_size: int,
    overlap: int,
    tile_batch_size: int,
) -> dict[str, float]:
    model.eval()
    intersection = prediction_pixels = target_pixels = union = 0.0
    progress = tqdm(samples, desc="Full-resolution validation", leave=False)
    for sample_dir in progress:
        image = load_rgb(image_path_for(sample_dir))
        target = np.load(
            mask_cache_dir / f"{sample_dir.name}.npy", allow_pickle=False
        ).astype(bool)
        probability = predict_full_resolution(
            model, image, device, patch_size, overlap, tile_batch_size
        )
        prediction = probability >= 0.5
        intersection += np.logical_and(prediction, target).sum()
        prediction_pixels += prediction.sum()
        target_pixels += target.sum()
        union += np.logical_or(prediction, target).sum()

    epsilon = 1e-7
    return {
        "dice": float(
            (2 * intersection + epsilon)
            / (prediction_pixels + target_pixels + epsilon)
        ),
        "iou": float((intersection + epsilon) / (union + epsilon)),
    }


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    if not history:
        return
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="Dice")
    axes[1].plot(epochs, [row["val_iou"] for row in history], label="IoU")
    axes[1].set(title="Original-resolution validation", xlabel="Epoch", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)


def save_checkpoint(checkpoint: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)


def validate_arguments(args: argparse.Namespace) -> None:
    if args.patch_size < 64 or args.patch_size % 16 != 0:
        raise ValueError("--patch-size must be at least 64 and divisible by 16")
    if args.tile_overlap < 0 or args.tile_overlap >= args.patch_size:
        raise ValueError("--tile-overlap must be between 0 and patch-size - 1")
    if args.batch_size < 1 or args.validation_tile_batch < 1:
        raise ValueError("Batch sizes must be positive")
    if args.patches_per_image < 1:
        raise ValueError("--patches-per-image must be positive")
    if not 0 <= args.positive_crop_probability <= 1:
        raise ValueError("--positive-crop-probability must be between 0 and 1")


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install CUDA-enabled PyTorch and verify "
            "torch.cuda.is_available() before training."
        )
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Source image resizing: DISABLED")
    print(f"Training patch size: {args.patch_size}x{args.patch_size}")

    samples = collect_samples(args.data_dir)
    if args.max_samples is not None:
        if args.max_samples < 10:
            raise ValueError("--max-samples must be at least 10")
        samples = samples[: args.max_samples]
    train_samples, validation_samples, test_samples = split_samples(
        samples, args.seed
    )
    print(
        f"Samples: train={len(train_samples)}, validation={len(validation_samples)}, "
        f"reserved test={len(test_samples)}"
    )

    split_manifest = {
        "seed": args.seed,
        "train": [sample.name for sample in train_samples],
        "validation": [sample.name for sample in validation_samples],
        "test": [sample.name for sample in test_samples],
    }
    with (args.output_dir / "splits.json").open("w", encoding="utf-8") as file:
        json.dump(split_manifest, file, indent=2)

    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir.resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    config["resume"] = str(args.resume.resolve()) if args.resume else None
    config["resizing"] = False
    config["validation_mode"] = "overlapping tiles at original resolution"
    with (args.output_dir / "training_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=2)

    mask_cache_dir = args.output_dir / "combined_masks_original_resolution"
    build_mask_cache(samples, mask_cache_dir)

    train_dataset = RandomPatchDataset(
        train_samples,
        mask_cache_dir,
        args.patch_size,
        args.patches_per_image,
        args.positive_crop_probability,
        augment=True,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = UNet(args.base_channels).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    scaler = create_grad_scaler(enabled=True)

    start_epoch = 1
    best_val_dice = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_config = checkpoint["config"]
        for field in ("patch_size", "base_channels"):
            if saved_config[field] != getattr(args, field):
                raise ValueError(
                    f"Resume mismatch for {field}: checkpoint={saved_config[field]}, "
                    f"command={getattr(args, field)}"
                )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_dice = checkpoint["best_val_dice"]
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        history = checkpoint.get("history", [])
        print(f"Resuming after epoch {checkpoint['epoch']}")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = training_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        validation_metrics = validate_full_images(
            model,
            validation_samples,
            mask_cache_dir,
            device,
            args.patch_size,
            args.tile_overlap,
            args.validation_tile_batch,
        )
        scheduler.step(validation_metrics["dice"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_dice": validation_metrics["dice"],
            "val_iou": validation_metrics["iou"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - start_time) / 60,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {row['train_loss']:.4f}, "
            f"Dice {row['train_dice']:.4f} | full-res val Dice "
            f"{row['val_dice']:.4f}, IoU {row['val_iou']:.4f}"
        )

        improved = validation_metrics["dice"] > best_val_dice
        if improved:
            best_val_dice = validation_metrics["dice"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": best_val_dice,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "config": config,
        }
        save_checkpoint(checkpoint, args.output_dir / "last_model.pt")
        if improved:
            save_checkpoint(checkpoint, args.output_dir / "best_model.pt")
            print(f"  Saved new best model (full-res Dice={best_val_dice:.4f})")
        save_history(history, args.output_dir)

        if (
            args.early_stopping > 0
            and epochs_without_improvement >= args.early_stopping
        ):
            print(
                f"Early stopping: no validation improvement for "
                f"{args.early_stopping} epochs."
            )
            break

    summary = {
        "completed_epochs": history[-1]["epoch"] if history else start_epoch - 1,
        "best_validation_dice": best_val_dice,
        "training_minutes_this_run": (time.time() - start_time) / 60,
        "source_images_resized": False,
        "patch_size": args.patch_size,
        "reserved_test_samples": len(test_samples),
    }
    with (args.output_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    print("\nTraining stage complete. The reserved test split was not evaluated.")
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
