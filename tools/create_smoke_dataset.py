"""Create a deterministic tiny dataset for local end-to-end validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("smoke_data"))
    parser.add_argument("--samples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Smoke dataset directory is not empty: {args.output_dir}")
    generator = np.random.default_rng(123)
    yy, xx = np.mgrid[:64, :64]
    for sample_index in range(args.samples):
        sample_id = f"synthetic-{sample_index:02d}"
        image_dir = args.output_dir / sample_id / "images"
        mask_dir = args.output_dir / sample_id / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        image = generator.normal(18, 3, size=(64, 64)).astype(np.float32)
        centers = [(15 + sample_index % 4, 16), (43, 42 - sample_index % 3)]
        if sample_index % 2:
            centers.append((18, 46))
        for instance_index, (row, column) in enumerate(centers, start=1):
            radius = 5 + (sample_index + instance_index) % 3
            mask = (yy - row) ** 2 + (xx - column) ** 2 <= radius**2
            image[mask] += 155 + 10 * instance_index
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                mask_dir / f"instance-{instance_index:02d}.png"
            )
        image = np.clip(image, 0, 255).astype(np.uint8)
        Image.fromarray(np.repeat(image[..., None], 3, axis=2)).save(image_dir / f"{sample_id}.png")


if __name__ == "__main__":
    main()
