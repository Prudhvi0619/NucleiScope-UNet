# Engineering notes

NucleiScope-UNet is designed so that a training run, its data, and its evaluation
artifacts can be traced and checked as one experiment.

## Data integrity

- Dataset manifests store SHA-256 hashes for every selected image and annotation.
- Annotation masks must be nonempty, unique, nonoverlapping, and contain one connected
  nucleus.
- Mask-cache entries are keyed by annotation content and validated before reuse.
- Higher-bit-depth inputs are normalized without silently saturating their dynamic range.
- Multipage TIFF inputs are rejected with instructions to export a 2-D slice or projection.

## Reproducible training

- Split generation is deterministic and approximately balances nucleus count, foreground
  fraction, and image area.
- Split and training seeds are independent.
- Python, NumPy, PyTorch CPU, and CUDA random states are saved in each checkpoint.
- Deterministic PyTorch algorithms are enabled by default.
- Resume validates the dataset, split, source, model, sampling, optimization, and inference
  contract before writing output metadata.
- The original configuration remains immutable; resume events are append-only.
- Positive training crops sample nucleus instances uniformly instead of favoring large nuclei.
- Image and mask padding use the same constant-background geometry.

## Inference and calibration

- Full images are reconstructed from overlapping tiles using tapered center weights.
- Semantic threshold, connected-component cleanup, and watershed settings are calibrated
  only on validation images.
- Calibration files store the checkpoint SHA-256 and cannot be applied to another model.
- Prediction directories include the input image hash to prevent same-name collisions.
- Instance-label TIFFs use signed 32-bit labels and validate label capacity.
- Watershed scale transfer must be declared explicitly; recalibration is preferred after a
  change in magnification, stain, microscope, or acquisition domain.

## Evaluation

- Checkpoint selection uses macro per-image validation Dice.
- Reports include semantic overlap and classification metrics, IoU-matched instance
  precision/recall/F1, mean AP across IoU 0.50–0.95, and counting error.
- Test summaries include bootstrap 95% confidence intervals.
- `run_multi_seed.py` repeats training seeds on one fixed split and creates aggregate
  mean and standard-deviation results.
- Unit and regression tests plus Ruff checks run on every push and pull request.

## Evidence boundary

The compact artifacts currently committed under `assets/` and `results/` come from the
reference benchmark. Its original checkpoint, dataset fingerprint, split manifest, and exact
environment were not committed, so those values cannot be retroactively upgraded to the
current provenance contract.

Run `train.py` and `test.py` with the current code to generate a new fingerprinted result
set. A separate labelled dataset is required before making external-domain generalization
claims.

