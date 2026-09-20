"""Run repeated training seeds on one fixed split and aggregate headline metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from nuclei_io import atomic_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated-seed NucleiScope benchmark")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multi_seed"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        run_dir = args.output_dir / f"seed-{seed}"
        evaluation_dir = run_dir / "evaluation"
        run(
            [
                sys.executable,
                "train.py",
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(run_dir),
                "--epochs",
                str(args.epochs),
                "--patch-size",
                str(args.patch_size),
                "--batch-size",
                str(args.batch_size),
                "--base-channels",
                str(args.base_channels),
                "--seed",
                str(seed),
                "--split-seed",
                str(args.split_seed),
                "--device",
                args.device,
            ]
        )
        run(
            [
                sys.executable,
                "test.py",
                "--data-dir",
                str(args.data_dir),
                "--checkpoint",
                str(run_dir / "best_model.pt"),
                "--output-dir",
                str(evaluation_dir),
                "--device",
                args.device,
                "--seed",
                str(args.split_seed),
            ]
        )
        with (evaluation_dir / "test_summary.json").open("r", encoding="utf-8") as file:
            summary = json.load(file)
        rows.append(
            {
                "training_seed": seed,
                "split_seed": args.split_seed,
                "mean_dice": summary["segmentation"]["macro_per_image"]["dice"]["mean"],
                "mean_instance_ap": summary["instance_segmentation"]["macro_per_image"][
                    "instance_mean_ap_0.50_0.95"
                ],
                "watershed_count_mae": summary["counting"]["test_watershed"]["mae"],
            }
        )
    with (args.output_dir / "multi_seed_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "standard_deviation": float(np.std([row[key] for row in rows], ddof=1)),
        }
        for key in ("mean_dice", "mean_instance_ap", "watershed_count_mae")
    }
    atomic_json_dump(
        {"runs": rows, "aggregate": aggregate}, args.output_dir / "multi_seed_summary.json"
    )


if __name__ == "__main__":
    main()
