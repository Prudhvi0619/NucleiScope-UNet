# Reproducibility contract

Every version-2 checkpoint contains:

- SHA-256 fingerprint of every selected image and annotation;
- full train/validation/test membership and split fingerprint;
- source-code fingerprint;
- immutable training contract;
- original configuration and an append-only resume record;
- Python, NumPy, CPU Torch, and CUDA RNG states;
- model, optimizer, scheduler, scaler, and complete history.

Resume validation occurs before any split or configuration metadata is rewritten. Patch
sampling is a deterministic function of training seed, epoch, and dataset index. DataLoader
workers are recreated each epoch, so a resumed run does not depend on hidden persistent-worker
state.

`test.py` recalculates the dataset fingerprint and uses the split embedded in the checkpoint.
It writes `calibration.json` containing the checkpoint SHA-256. `predict.py` refuses a
calibration artifact whose hash does not match the supplied checkpoint.

For a defensible report:

1. Preserve `dataset_manifest.json`, `splits.json`, `training_config.json`, `history.csv`,
   `best_model.pt`, `last_model.pt`, evaluation CSV/JSON files, and dependency lock files.
2. Run at least three training seeds with one fixed split.
3. Report mean, standard deviation, and bootstrap confidence intervals.
4. Keep the held-out test split untouched until all choices are finalized.
5. Evaluate an external microscopy dataset before claiming domain generalization.

Version-1 or incompatible checkpoints are intentionally not loaded through unsafe pickle
fallback. Retrain them with the current pipeline instead.
