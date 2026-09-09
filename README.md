# NucleiScope-UNet

Full-resolution microscopy nuclei segmentation and quantification using a
patch-wise U-Net. The pipeline trains and predicts with pixel-preserving crops
and overlapping tiles; source images and masks are never geometrically resized.

## Highlights

- Trains on 670 annotated microscopy images with nine native image sizes.
- Extracts nucleus-aware 256 x 256 patches without changing spatial scale.
- Reconstructs full-resolution masks using overlapping tiled inference.
- Separates touching nuclei with validation-calibrated watershed post-processing.
- Reports semantic-segmentation and nucleus-counting metrics on an untouched
  test split.
- Produces masks, instance labels, overlays, probability maps, and JSON
  quantification reports for unseen images.

## Results

The model was evaluated once on a held-out 67-image test set. Model selection
and watershed calibration used validation data only.

| Metric | Test result |
|---|---:|
| Mean Dice | 0.870 |
| Median Dice | 0.932 |
| Pooled Dice | 0.888 |
| Mean IoU | 0.799 |
| Mean precision | 0.905 |
| Mean recall | 0.873 |
| Mean pixel accuracy | 97.43% |
| Mean MCC | 0.863 |
| Watershed count MAE | 6.12 nuclei |
| Watershed median absolute error | 3 nuclei |
| Count Pearson correlation | 0.967 |

Watershed separation improved count MAE from **7.84** using connected
components alone to **6.12** nuclei.

![Training curves](assets/training_curves.png)

![Segmentation metrics](assets/segmentation_metrics.png)

![Counting metrics](assets/counting_metrics.png)

![Representative predictions](assets/representative_predictions.png)

## Method

1. Merge the instance annotations for each image into a binary semantic mask.
2. Split complete images into deterministic 80/10/10 train, validation, and
   test subsets before extracting patches.
3. Sample pixel-preserving 256 x 256 training crops, with 75% of crops centred
   near foreground pixels.
4. Train a U-Net with BCE + soft Dice loss, AdamW, mixed precision, learning-rate
   reduction, and early stopping.
5. Run overlapping-tile inference with 64-pixel overlap and average predictions
   back into the native image canvas.
6. Select watershed parameters using validation images only and report final
   segmentation and counting results on the untouched test set.

The best checkpoint occurred at epoch 15 with validation Dice 0.9063. Training
stopped at epoch 23 after eight epochs without improvement.

## Repository structure

```text
NucleiScope-UNet/
|-- train.py                  # Patch-wise training and full-resolution validation
|-- test.py                   # Held-out segmentation and counting evaluation
|-- predict.py                # Prediction and quantification for one image
|-- postprocess.py            # Component cleanup and watershed separation
|-- requirements.txt
|-- assets/                   # Selected result figures
|-- results/                  # Compact JSON/CSV evaluation artifacts
|-- .gitignore
|-- LICENSE
`-- README.md
```

The dataset, mask cache, bulk prediction masks, and model checkpoints are
excluded from version control.

## Dataset layout

Download the [2018 Data Science Bowl dataset](https://www.kaggle.com/competitions/data-science-bowl-2018)
and keep the original `stage1_train` structure:

```text
data/stage1_train/
`-- <sample_id>/
    |-- images/<sample_id>.png
    `-- masks/<one PNG per nucleus>
```

## Installation

Python 3.10 or newer is recommended. Install a CUDA-enabled PyTorch build that
matches the system first, then install the remaining packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Use the correct PyTorch CUDA index for the installed driver if it differs from
the example above.

## Training

```bash
python train.py --data-dir "data/stage1_train" --output-dir outputs --epochs 30 --patch-size 256 --batch-size 4 --patches-per-image 2
```

For a 6 GB GPU, reduce `--batch-size` to 2 if an out-of-memory error occurs.
Resume an interrupted run with:

```bash
python train.py --data-dir "data/stage1_train" --output-dir outputs --epochs 30 --resume "outputs/last_model.pt"
```

## Testing

```bash
python test.py --data-dir "data/stage1_train" --checkpoint "outputs/best_model.pt" --output-dir "outputs/test"
```

The evaluator saves per-image CSV files, a machine-readable summary, semantic
masks, watershed instance masks, and metric visualizations.

## Prediction

After testing has produced the calibrated post-processing settings:

```bash
python predict.py --image "path/to/microscopy_image.png" --checkpoint "outputs/best_model.pt"
```

Prediction outputs include the raw and cleaned semantic masks, a 16-bit
probability map, watershed instance labels, coloured instances, an overlay, a
summary figure, and `quantification.json`.

![Prediction example](assets/prediction_example.png)

## Limitations

- One brightfield-style test image was a strong domain outlier (Dice 0.033),
  while the median test Dice was 0.932. Performance depends on the microscopy
  appearance represented during training.
- Watershed improves touching-nucleus separation but remains a heuristic, not
  a learned instance-segmentation model.
- Areas and equivalent diameters are reported in pixels. Physical units require
  microscope pixel-size calibration.
- Nuclei per megapixel is image-space density and must not be interpreted as a
  physical density measurement.

## License

Released under the MIT License.
