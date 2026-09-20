# Model card

## Model description

NucleiScope uses a four-level U-Net for binary 2-D nucleus segmentation. Training samples
native-resolution patches. Inference blends overlapping tiles, then removes small components
and applies marker-controlled watershed for approximate instance separation.

## Intended use

- Educational and portfolio demonstrations of microscopy segmentation.
- Research prototyping on imagery whose appearance and physical scale are represented in
  the calibration data.
- Human-reviewed quantitative exploration, not autonomous clinical decisions.

## Out-of-scope use

- Diagnosis, treatment selection, or safety-critical laboratory decisions.
- 3-D volumes, multipage TIFF stacks, or time series without an explicit 2-D conversion.
- Uncalibrated transfer across magnifications or microscopy domains.

## Training data

The intended dataset is the 2018 Data Science Bowl `stage1_train` collection containing 670
annotated images. The repository does not redistribute the images. Every run records file
hashes and rejects mismatched data at evaluation time.

## Evaluation

The repaired evaluator reports semantic overlap/classification metrics, object matching at
IoU 0.50, mean instance AP across IoU 0.50–0.95, count error, per-image records, and bootstrap
confidence intervals. Model selection uses validation macro Dice. Semantic and post-processing
settings are selected on validation data only.

The CSV/JSON files currently committed under `results/` came from the legacy implementation.
They do not contain enough provenance to qualify as verified repaired-pipeline results.

## Known risks

- Watershed is a hand-designed approximation and can merge or split nuclei.
- Fixed pixel-scale parameters do not automatically transfer across magnification.
- Rare acquisition styles can fail even when aggregate metrics are strong.
- A high Pearson count correlation does not imply low per-image count error.
- The repository contains no trained checkpoint, so inference requires retraining.

## Reproducibility

Training defaults to deterministic algorithms and stores the dataset, split, code fingerprints,
configuration, RNG state, optimizer, scheduler, scaler, and history. Exact determinism still
depends on using compatible hardware, drivers, PyTorch, and pinned dependencies.
