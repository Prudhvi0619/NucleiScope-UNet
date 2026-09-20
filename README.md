# NucleiScope-UNet

Reproducible, native-resolution nuclei segmentation and quantification using a
patch-wise U-Net, validation-calibrated post-processing, and object-level evaluation.

This is the repaired pipeline. It keeps source pixels at their original spatial scale,
supports CPU smoke runs and CUDA training, and treats the dataset, split, checkpoint,
calibration, and source revision as one verifiable experiment.

## What is improved

- Content hashes bind every checkpoint to the exact dataset and split.
- Resume rejects changed hyperparameters, data, or splits before writing metadata.
- Random, NumPy, PyTorch, and CUDA state is saved; deterministic execution is the default.
- Masks are cached with annotation hashes and structural validation.
- Nucleus-aware patches sample instances uniformly rather than favoring large nuclei.
- 16-bit images preserve dynamic range; multipage TIFF stacks are rejected explicitly.
- CPU execution is supported for tests and small runs.
- Overlapping predictions use tapered blending to reduce tile-border artifacts.
- Semantic threshold, connected-components cleanup, and watershed settings are calibrated
  independently using validation data.
- Evaluation includes semantic metrics, instance AP/F1, counting metrics, bootstrap
  confidence intervals, and per-image calibration records.
- Prediction refuses calibration from a different checkpoint.
- Automated tests, linting, and GitHub Actions are included.

## Current evidence status

The repository does **not** contain the original dataset, trained weights, split manifest,
or full history. Therefore the historical numbers in `results/` cannot independently prove
which exact artifacts generated them. They remain a legacy baseline from the earlier
pipeline and are not presented as results from this repaired implementation.

The earlier single-run baseline reported mean Dice `0.870`, pooled Dice `0.888`, and
watershed count MAE `6.12` on 67 test images. It also had a substantial failure tail and
29.31% count MAPE. Re-run this version before using those values in a resume or interview.

## Requirements

Python **3.10 or newer is required**. The fully resolved CPU lock file and GitHub
Actions workflow use **Python 3.12** so that every transitive version is reproducible.

CPU installation:

```bash
# Use Python 3.12 with the exact lock file.
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements-lock-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

For CUDA, install the PyTorch `2.4.1` build matching the machine first, then install the
remaining pinned dependencies:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The CUDA index is an example; use the index compatible with the installed driver.

## Dataset

Download the [2018 Data Science Bowl data](https://www.kaggle.com/competitions/data-science-bowl-2018)
and retain its instance-mask layout:

```text
data/stage1_train/
└── <sample_id>/
    ├── images/<sample_id>.png
    └── masks/<one lossless image per nucleus>
```

Each mask must be nonempty, contain exactly one connected nucleus, have the same dimensions
as its image, and not overlap another instance mask. Lossy JPEG annotations are discouraged.

## Train

```bash
python train.py \
  --data-dir data/stage1_train \
  --output-dir outputs/run-001 \
  --epochs 30 \
  --patch-size 256 \
  --batch-size 4
```

The default split is approximately stratified by nucleus count, foreground fraction, and
image area. It reserves complete images before patch sampling. `--split-seed` controls the
split independently from `--seed`, which controls training.

Resume by repeating all original training arguments and increasing `--epochs`:

```bash
python train.py \
  --data-dir data/stage1_train \
  --output-dir outputs/run-001 \
  --epochs 40 \
  --resume outputs/run-001/last_model.pt
```

Changed data or training settings are rejected. A new run also refuses a nonempty output
directory, preventing stale checkpoint/configuration mixtures.

## Evaluate and calibrate

```bash
python test.py \
  --data-dir data/stage1_train \
  --checkpoint outputs/run-001/best_model.pt \
  --output-dir outputs/run-001-evaluation
```

The evaluator predicts validation images first, selects the semantic threshold and counting
settings without test access, then evaluates the held-out test split once. It writes:

- `calibration.json`, bound to the SHA-256 of the checkpoint;
- semantic, instance, and counting metrics per image;
- independent connected-component and watershed searches;
- 95% bootstrap confidence intervals;
- 32-bit instance-label TIFFs;
- corrected overlays that tint only predicted foreground.

Use `--threshold` only when the value was declared before test evaluation.

## Predict

Prediction requires either a verified calibration artifact or every setting explicitly:

```bash
python predict.py \
  --image path/to/image.tiff \
  --checkpoint outputs/run-001/best_model.pt \
  --calibration outputs/run-001-evaluation/calibration.json
```

The calibration checkpoint hash must match. Output folders include the input file hash, so
different images with the same filename cannot overwrite one another.

Watershed settings are measured in pixels. If an image has a known linear pixel scale ratio
relative to calibration, pass `--postprocess-scale-factor`. Prefer recalibration whenever
the microscope, magnification, stain, or acquisition domain changes.

## Repeated seeds

Run three training seeds on one fixed split:

```bash
python run_multi_seed.py \
  --data-dir data/stage1_train \
  --output-dir outputs/multi-seed \
  --seeds 41 42 43 \
  --split-seed 42
```

This produces per-run and aggregate mean/standard-deviation results. A separate external
dataset is still needed to establish cross-domain generalization.

## Quality checks

```bash
pip install -r requirements-lock-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
ruff check .
pytest
```

CI runs these checks on every push and pull request.

## Repository structure

```text
├── train.py                 # deterministic training and tiled inference
├── test.py                  # calibration plus semantic/instance/count evaluation
├── predict.py               # checkpoint-bound prediction and quantification
├── postprocess.py           # cleanup and watershed separation
├── nuclei_io.py             # validated I/O, hashing, and mask cache
├── run_multi_seed.py        # repeated-seed benchmark runner
├── tests/                   # unit and regression tests
├── requirements-lock-cpu.txt # fully resolved reference environment
├── results/                 # explicitly labelled legacy baseline artifacts
├── MODEL_CARD.md
├── REPRODUCIBILITY.md
└── .github/workflows/quality.yml
```

## Scope and limitations

- This is a 2-D semantic model followed by heuristic instance separation, not a learned
  instance-segmentation architecture.
- TIFF stacks must be converted into an explicit slice or projection.
- Performance is not established outside the heterogeneous Data Science Bowl data.
- Pixel measurements are not physical measurements without microscope calibration.
- The trained checkpoint is not included; publish a verified release artifact before
  presenting the repository as immediately usable for inference.

See [MODEL_CARD.md](MODEL_CARD.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before reuse.

## License

Released under the MIT License.
