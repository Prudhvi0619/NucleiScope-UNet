"""Train a reproducible patch-wise U-Net on native-resolution microscopy images."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
    load_rgb,
    sha256_file,
    sha256_json,
)

CHECKPOINT_SCHEMA_VERSION = 2
CONTRACT_FIELDS = (
    "seed",
    "patch_size",
    "patches_per_image",
    "positive_crop_probability",
    "tile_overlap",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "base_channels",
    "split_strategy",
    "split_seed",
    "deterministic",
    "max_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible native-resolution nuclei segmentation training"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage1_train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-image", type=int, default=2)
    parser.add_argument("--positive-crop-probability", type=float, default=0.75)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--validation-tile-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Keep fixed across repeated training seeds for the same split.",
    )
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split-strategy", choices=("stratified", "random"), default="stratified")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic algorithms. Disable only for an explicitly faster run.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
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
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if args.base_channels < 1:
        raise ValueError("--base-channels must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.early_stopping < 0:
        raise ValueError("--early-stopping cannot be negative")
    if args.max_samples is not None and args.max_samples < 10:
        raise ValueError("--max-samples must be at least 10")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def _jsonable_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value.resolve())
    values["source_images_resized"] = False
    values["normalization"] = (
        "8-bit unchanged; higher bit depth robustly scaled at 0.5/99.5 percentiles"
    )
    return values


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    records = []
    for name in ("train.py", "test.py", "predict.py", "postprocess.py", "nuclei_io.py"):
        path = root / name
        if path.is_file():
            records.append({"name": name, "sha256": sha256_file(path)})
    return sha256_json(records)


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": repr(random.getstate()),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(ast.literal_eval(state["python"]))
    value = state["numpy"]
    np.random.set_state(
        (
            value["bit_generator"],
            np.asarray(value["keys"], dtype=np.uint32),
            int(value["position"]),
            int(value["has_gauss"]),
            float(value["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "A PyTorch version supporting safe weights_only loading is required"
        ) from error
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Legacy or incompatible checkpoint. Retrain with this repaired pipeline; "
            "unsafe pickle fallback is intentionally disabled."
        )
    return checkpoint


def split_samples_random(
    samples: Sequence[Path], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    train_end = int(0.80 * len(shuffled))
    validation_end = int(0.90 * len(shuffled))
    return shuffled[:train_end], shuffled[train_end:validation_end], shuffled[validation_end:]


def split_samples_stratified(
    samples: Sequence[Path], metadata: dict[str, dict], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    """Approximately balance count, density, and image area across all splits."""
    rng = random.Random(seed)
    ordered = list(samples)
    rng.shuffle(ordered)
    ordered.sort(
        key=lambda sample: (
            metadata[sample.name]["instance_count"],
            metadata[sample.name]["foreground_fraction"],
            int(np.prod(metadata[sample.name]["shape"])),
        )
    )
    total = len(ordered)
    targets = {"train": int(0.80 * total), "validation": int(0.10 * total)}
    targets["test"] = total - targets["train"] - targets["validation"]
    result: dict[str, list[Path]] = {key: [] for key in targets}
    for sample in ordered:
        eligible = [key for key in targets if len(result[key]) < targets[key]]
        scores = {key: (targets[key] - len(result[key])) / max(targets[key], 1) for key in eligible}
        best_score = max(scores.values())
        candidates = [key for key in eligible if scores[key] == best_score]
        result[rng.choice(candidates)].append(sample)
    for values in result.values():
        rng.shuffle(values)
    return result["train"], result["validation"], result["test"]


def pad_to_patch(
    image: np.ndarray, mask: np.ndarray, patch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    pad_bottom = max(0, patch_size - height)
    pad_right = max(0, patch_size - width)
    if pad_bottom == 0 and pad_right == 0:
        return image, mask
    image = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="constant")
    mask = np.pad(mask, ((0, pad_bottom), (0, pad_right)), mode="constant")
    return image, mask


class RandomPatchDataset(Dataset):
    def __init__(
        self,
        samples,
        mask_cache_dir,
        patch_size,
        patches_per_image,
        positive_crop_probability,
        seed,
        augment=True,
    ) -> None:
        self.samples = list(samples)
        self.mask_cache_dir = mask_cache_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.positive_crop_probability = positive_crop_probability
        self.seed = seed
        self.augment = augment
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def _rng(self, index: int) -> random.Random:
        value = (self.seed * 1_000_003 + self.epoch * 97_409 + index * 65_537) & 0xFFFFFFFF
        return random.Random(value)

    def _crop_origin(self, mask, centers, rng) -> tuple[int, int]:
        patch = self.patch_size
        height, width = mask.shape
        if rng.random() < self.positive_crop_probability and len(centers):
            center_y, center_x = centers[rng.randrange(len(centers))]
            jitter = patch // 4
            center_y = int(round(center_y)) + rng.randint(-jitter, jitter)
            center_x = int(round(center_x)) + rng.randint(-jitter, jitter)
            top = max(0, min(center_y - patch // 2, height - patch))
            left = max(0, min(center_x - patch // 2, width - patch))
        else:
            top = rng.randint(0, max(0, height - patch))
            left = rng.randint(0, max(0, width - patch))
        return top, left

    def __getitem__(self, index: int):
        rng = self._rng(index)
        sample_dir = self.samples[index % len(self.samples)]
        image = load_rgb(image_path_for(sample_dir))
        mask = load_cached_mask(self.mask_cache_dir, sample_dir.name)
        centers = load_cached_centers(self.mask_cache_dir, sample_dir.name)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask size mismatch for {sample_dir.name}")
        image, mask = pad_to_patch(image, mask, self.patch_size)
        top, left = self._crop_origin(mask, centers, rng)
        image = image[top : top + self.patch_size, left : left + self.patch_size]
        mask = mask[top : top + self.patch_size, left : left + self.patch_size]
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        if self.augment:
            if rng.random() < 0.5:
                image_tensor, mask_tensor = (
                    torch.flip(image_tensor, (2,)),
                    torch.flip(mask_tensor, (2,)),
                )
            if rng.random() < 0.5:
                image_tensor, mask_tensor = (
                    torch.flip(image_tensor, (1,)),
                    torch.flip(mask_tensor, (1,)),
                )
            rotations = rng.randint(0, 3)
            if rotations:
                image_tensor = torch.rot90(image_tensor, rotations, (1, 2))
                mask_tensor = torch.rot90(mask_tensor, rotations, (1, 2))
            if rng.random() < 0.4:
                image_tensor = torch.clamp(
                    image_tensor * rng.uniform(0.90, 1.10) + rng.uniform(-0.05, 0.05), 0, 1
                )
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

    def forward(self, inputs):
        return self.layers(inputs)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, inputs):
        return self.layers(inputs)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs, skip):
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
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, inputs):
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

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum((1, 2, 3))
        denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
        return bce + (1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))).mean()


def _autocast(device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def training_epoch(model, loader, criterion, optimizer, scaler, device) -> dict[str, float]:
    model.train()
    total_loss = total_items = 0.0
    intersection = prediction_pixels = target_pixels = union = 0.0
    for images, masks in tqdm(loader, desc="Training patches", leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            logits = model(images)
            loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
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
    epsilon = 1e-7
    return {
        "loss": total_loss / max(total_items, 1),
        "dice": (2 * intersection + epsilon) / (prediction_pixels + target_pixels + epsilon),
        "iou": (intersection + epsilon) / (union + epsilon),
    }


def tile_origins(length: int, patch_size: int, overlap: int) -> list[int]:
    if patch_size < 1 or overlap < 0 or overlap >= patch_size:
        raise ValueError("Invalid patch size or overlap")
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
    model, image, device, patch_size, overlap, tile_batch_size
) -> np.ndarray:
    if tile_batch_size < 1:
        raise ValueError("tile_batch_size must be positive")
    original_height, original_width = image.shape[:2]
    padded_image, _ = pad_to_patch(
        image, np.zeros((original_height, original_width), np.uint8), patch_size
    )
    height, width = padded_image.shape[:2]
    y_origins = tile_origins(height, patch_size, overlap)
    x_origins = tile_origins(width, patch_size, overlap)
    one_dimensional = np.maximum(np.hanning(patch_size).astype(np.float32), 0.05)
    blending_window = np.outer(one_dimensional, one_dimensional)
    probability_sum = np.zeros((height, width), np.float32)
    contribution_weight = np.zeros((height, width), np.float32)
    tiles, locations = [], []

    def infer_pending() -> None:
        if not tiles:
            return
        batch = np.stack(tiles).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        with _autocast(device):
            probabilities = torch.sigmoid(model(tensor))[:, 0]
        for probability, (top, left) in zip(
            probabilities.float().cpu().numpy(), locations, strict=False
        ):
            region = np.s_[top : top + patch_size, left : left + patch_size]
            probability_sum[region] += probability * blending_window
            contribution_weight[region] += blending_window
        tiles.clear()
        locations.clear()

    for top in y_origins:
        for left in x_origins:
            tiles.append(padded_image[top : top + patch_size, left : left + patch_size])
            locations.append((top, left))
            if len(tiles) >= tile_batch_size:
                infer_pending()
    infer_pending()
    probability = probability_sum / np.maximum(contribution_weight, 1e-7)
    return probability[:original_height, :original_width]


@torch.inference_mode()
def validate_full_images(
    model, samples, mask_cache_dir, device, patch_size, overlap, tile_batch_size
):
    model.eval()
    per_image_dice, per_image_iou = [], []
    intersection = prediction_pixels = target_pixels = union = 0.0
    for sample_dir in tqdm(samples, desc="Full-resolution validation", leave=False):
        image = load_rgb(image_path_for(sample_dir))
        target = load_cached_mask(mask_cache_dir, sample_dir.name).astype(bool)
        prediction = (
            predict_full_resolution(model, image, device, patch_size, overlap, tile_batch_size)
            >= 0.5
        )
        item_intersection = np.logical_and(prediction, target).sum()
        item_prediction, item_target = prediction.sum(), target.sum()
        item_union = np.logical_or(prediction, target).sum()
        per_image_dice.append(
            (2 * item_intersection + 1e-7) / (item_prediction + item_target + 1e-7)
        )
        per_image_iou.append((item_intersection + 1e-7) / (item_union + 1e-7))
        intersection += item_intersection
        prediction_pixels += item_prediction
        target_pixels += item_target
        union += item_union
    return {
        "macro_dice": float(np.mean(per_image_dice)),
        "macro_iou": float(np.mean(per_image_iou)),
        "pooled_dice": float(
            (2 * intersection + 1e-7) / (prediction_pixels + target_pixels + 1e-7)
        ),
        "pooled_iou": float((intersection + 1e-7) / (union + 1e-7)),
    }


def save_history(history: list[dict], output_dir: Path) -> None:
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [row["val_macro_dice"] for row in history], label="Macro Dice")
    axes[1].plot(epochs, [row["val_pooled_dice"] for row in history], label="Pooled Dice")
    axes[1].set(title="Validation", xlabel="Epoch", ylim=(0, 1))
    axes[1].legend()
    axes[2].plot(epochs, [row["learning_rate"] for row in history])
    axes[2].set(title="Learning rate", xlabel="Epoch", yscale="log")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)


def save_checkpoint(checkpoint: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)


def _contract(args, dataset_fingerprint, current_source_fingerprint):
    values = {field: getattr(args, field) for field in CONTRACT_FIELDS}
    values["dataset_fingerprint"] = dataset_fingerprint
    values["source_fingerprint"] = current_source_fingerprint
    return values


def _validate_resume(checkpoint, contract, args):
    saved = checkpoint.get("run_contract", {})
    mismatches = {
        key: {"checkpoint": saved.get(key), "command": value}
        for key, value in contract.items()
        if saved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume contract mismatch: {json.dumps(mismatches, indent=2)}")
    if args.output_dir.resolve() != args.resume.resolve().parent:
        raise ValueError("A resumed run must write beside its checkpoint")
    if args.epochs <= int(checkpoint["epoch"]):
        raise ValueError("--epochs must be greater than the checkpoint epoch")


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    device = resolve_device(args.device)
    configure_reproducibility(args.seed, args.deterministic)
    samples = collect_samples(args.data_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    dataset_manifest = build_dataset_manifest(samples)
    dataset_fingerprint = dataset_manifest["dataset_fingerprint"]
    current_source_fingerprint = source_fingerprint()
    contract = _contract(args, dataset_fingerprint, current_source_fingerprint)

    resume_checkpoint = None
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_checkpoint = load_checkpoint(args.resume, device)
        _validate_resume(resume_checkpoint, contract, args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        ensure_new_output_directory(args.output_dir)

    mask_cache_dir = args.output_dir / "combined_masks_original_resolution"
    cache_metadata = build_mask_cache(samples, mask_cache_dir, dataset_manifest)
    samples_by_id = {sample.name: sample for sample in samples}
    if resume_checkpoint is not None:
        split_manifest = resume_checkpoint["split_manifest"]
        all_ids = split_manifest["train"] + split_manifest["validation"] + split_manifest["test"]
        if set(all_ids) != set(samples_by_id):
            raise ValueError("Checkpoint split does not exactly match the fingerprinted dataset")
        train_samples = [samples_by_id[value] for value in split_manifest["train"]]
        validation_samples = [samples_by_id[value] for value in split_manifest["validation"]]
        test_samples = [samples_by_id[value] for value in split_manifest["test"]]
    else:
        if args.split_strategy == "stratified":
            train_samples, validation_samples, test_samples = split_samples_stratified(
                samples, cache_metadata, args.split_seed
            )
        else:
            train_samples, validation_samples, test_samples = split_samples_random(
                samples, args.split_seed
            )
        split_manifest = {
            "schema_version": 2,
            "strategy": args.split_strategy,
            "seed": args.split_seed,
            "dataset_fingerprint": dataset_fingerprint,
            "train": [sample.name for sample in train_samples],
            "validation": [sample.name for sample in validation_samples],
            "test": [sample.name for sample in test_samples],
        }
        split_manifest["split_fingerprint"] = sha256_json(split_manifest)

    config = _jsonable_args(args)
    config["dataset_fingerprint"] = dataset_fingerprint
    config["source_fingerprint"] = current_source_fingerprint
    atomic_json_dump(dataset_manifest, args.output_dir / "dataset_manifest.json")
    atomic_json_dump(split_manifest, args.output_dir / "splits.json")
    if resume_checkpoint is None:
        original_config = config
        resume_events = []
    else:
        original_config = resume_checkpoint["original_config"]
        resume_events = list(resume_checkpoint.get("resume_events", []))
        resume_events.append(
            {"from_epoch": int(resume_checkpoint["epoch"]), "target_epoch": args.epochs}
        )
    atomic_json_dump(
        {"original_config": original_config, "resume_events": resume_events},
        args.output_dir / "training_config.json",
    )

    train_dataset = RandomPatchDataset(
        train_samples,
        mask_cache_dir,
        args.patch_size,
        args.patches_per_image,
        args.positive_crop_probability,
        args.seed,
    )
    model = UNet(args.base_channels).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    scaler = create_grad_scaler(enabled=device.type == "cuda")
    start_epoch, best_val_dice, epochs_without_improvement = 1, -1.0, 0
    history = []
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_val_dice = float(resume_checkpoint["best_val_dice"])
        epochs_without_improvement = int(resume_checkpoint["epochs_without_improvement"])
        history = list(resume_checkpoint["history"])
        restore_rng_state(resume_checkpoint["rng_state"])

    print(f"Device: {device}")
    print(f"Dataset fingerprint: {dataset_fingerprint}")
    print(
        f"Samples: {len(train_samples)} train, {len(validation_samples)} validation, {len(test_samples)} test"
    )
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        train_metrics = training_epoch(model, train_loader, criterion, optimizer, scaler, device)
        validation_metrics = validate_full_images(
            model,
            validation_samples,
            mask_cache_dir,
            device,
            args.patch_size,
            args.tile_overlap,
            args.validation_tile_batch,
        )
        scheduler.step(validation_metrics["macro_dice"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_macro_dice": validation_metrics["macro_dice"],
            "val_macro_iou": validation_metrics["macro_iou"],
            "val_pooled_dice": validation_metrics["pooled_dice"],
            "val_pooled_iou": validation_metrics["pooled_iou"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        improved = validation_metrics["macro_dice"] > best_val_dice
        if improved:
            best_val_dice, epochs_without_improvement = validation_metrics["macro_dice"], 0
        else:
            epochs_without_improvement += 1
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": best_val_dice,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "original_config": original_config,
            "resume_events": resume_events,
            "run_contract": contract,
            "split_manifest": split_manifest,
            "dataset_fingerprint": dataset_fingerprint,
            "source_fingerprint": config["source_fingerprint"],
            "rng_state": capture_rng_state(),
        }
        save_checkpoint(checkpoint, args.output_dir / "last_model.pt")
        if improved:
            save_checkpoint(checkpoint, args.output_dir / "best_model.pt")
        save_history(history, args.output_dir)
        print(
            f"Epoch {epoch:02d}: loss={row['train_loss']:.4f}, macro val Dice={row['val_macro_dice']:.4f}, pooled val Dice={row['val_pooled_dice']:.4f}"
        )
        if args.early_stopping and epochs_without_improvement >= args.early_stopping:
            print(f"Early stopping after {epochs_without_improvement} unimproved epochs")
            break

    summary = {
        "schema_version": 2,
        "completed_epochs": int(history[-1]["epoch"]),
        "best_macro_validation_dice": best_val_dice,
        "training_minutes_this_session": (time.time() - start_time) / 60,
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_manifest["split_fingerprint"],
        "source_fingerprint": config["source_fingerprint"],
        "deterministic": args.deterministic,
        "device": str(device),
        "reserved_test_samples": len(test_samples),
    }
    atomic_json_dump(summary, args.output_dir / "training_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
