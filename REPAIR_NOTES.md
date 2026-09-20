# Repair notes

This edition was built from the earlier GitHub-ready folder without modifying it.

## Corrected

| Earlier problem | Resolution |
|---|---|
| Resume checked only patch size and channels | Immutable resume contract now includes dataset fingerprint, seeds, split strategy, patch sampling, optimization, and inference geometry. |
| Split/config overwritten before resume validation | Checkpoint and dataset are verified before experiment metadata is written. |
| Resume lost original provenance | Original configuration is immutable; resume events are append-only. |
| Nondeterministic cuDNN and missing RNG state | Deterministic algorithms are default; Python, NumPy, Torch, and CUDA states are saved and restored. Patch sampling is epoch/index deterministic. |
| Stale mask cache | Cache entries are keyed by annotation content hashes and validated for schema, dtype, shape, values, and instance count. Corrupt entries rebuild. |
| Windows could not delete an open memory map | Cache validation no longer memory-maps then deletes the same file; replacements are atomic. |
| 16-bit images saturated during RGB conversion | Higher-bit-depth inputs are robustly normalized before conversion to three channels. |
| TIFF stacks silently lost frames | Multipage inputs fail with an explicit instruction to export a slice or projection. |
| Checkpoint, dataset, and split were not linked | SHA-256 dataset, split, source, checkpoint, and calibration identities are recorded and checked. |
| Unsafe checkpoint loading | Only version-2 checkpoints using `weights_only=True` are accepted. |
| CUDA-only execution | `auto`, `cuda`, and `cpu` device modes are supported. |
| Minimum dependency versions | Runtime, CPU, and development dependencies are exactly pinned. |
| No tests or CI | Unit/regression tests, Ruff checks, and GitHub Actions were added. |
| Foreground-pixel sampling favored large nuclei | Positive crops select cached nucleus instances uniformly. |
| Random unstratified split | Default splitting approximately balances instance count, foreground fraction, and image area. Split and training seeds are independent. |
| Model selection used pooled validation Dice | Macro per-image Dice selects checkpoints; pooled metrics are retained for transparency. |
| Fixed uncalibrated semantic threshold | The threshold is selected by validation macro Dice unless explicitly predeclared. |
| Watershed and connected-components shared one cleanup value | Each method is calibrated independently on validation images. |
| No instance-quality evaluation | IoU-matched instance precision, recall, F1, and mean AP over IoU 0.50–0.95 are reported. |
| No uncertainty estimate | Test summaries include bootstrap 95% confidence intervals. |
| Only one seed was convenient | `run_multi_seed.py` repeats training seeds on one fixed split and aggregates results. |
| Test tile batch could be zero | Tiling validates positive batch size in the shared inference function and every CLI. |
| Old outputs could mix with new runs | Training/evaluation require empty output directories; prediction folders include input hashes. |
| Same-stem inputs overwrote predictions | Image SHA-256 is part of the output folder name. |
| 16-bit instance-label overflow | Instance TIFFs use signed 32-bit labels and check capacity. |
| Overlay tinted the entire image | Overlay blending is applied only where the prediction is foreground. |
| Small-image padding reflected nuclei into background labels | Image and mask now receive consistent constant-background padding. |
| Uniform tile averaging amplified border artifacts | Tapered overlap weights emphasize tile centers. |
| Ground-truth count trusted filenames | Annotation masks are validated as nonempty, unique, nonoverlapping, single connected instances. |
| Pixel-scale watershed settings silently transferred | Prediction records the scale and supports an explicit linear scale factor; recalibration is recommended. |
| Calibration could belong to another model | Prediction verifies the calibration's checkpoint SHA-256. |
| Python requirement was understated | Documentation states Python 3.10+ is required. |

## Evidence that cannot be reconstructed

The original dataset, checkpoint, split manifest, full history, and exact environment were
not present in or around the source repository. Therefore the old metrics cannot be converted
into version-2 provenance after the fact. They are labelled as legacy artifacts.

The repaired code must be retrained to obtain current metrics, repeated-seed variance, and a
publishable checkpoint. External-domain performance also requires a separate labelled dataset;
code changes cannot substitute for that evidence.
