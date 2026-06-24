# Supplementary Code README

## 1. Overview
This supplementary package contains the code used in our study on CT-based multiclass diagnosis of inner ear malformations (IEMs). The package includes three components:

1. `preprocess.ipynb`: image preprocessing and ROI extraction
2. `TransMIL_main.py`: model training and inference
3. `gradcam.py`: Grad-CAM visualization for explainability


These materials are provided to facilitate methodological review and reproducibility assessment during peer review.

## 2. Included files

### `preprocess.ipynb`
This Jupyter notebook contains the main preprocessing workflow:
- batch conversion from DICOM to NIfTI (`.nii.gz`)
- voxel spacing resampling
- YOLOX-based localization of the inner ear region of interest (ROI)
- cropping and export of left/right inner ear ROI volumes

The notebook is organized as a step-by-step workflow and includes a single-case example at the end.

### `TransMIL-main.py`
This is the main script for model training and inference. It includes:
- loading of ROI-based NIfTI volumes
- uniform sampling of 30 slices per case
- slice resizing and normalization
- patient-level train/validation splitting
- an improved TransMIL-based architecture for multiclass classification
- model training, validation, testing, prediction export, and top-K slice extraction

Main architectural components implemented in this script include:
- shared ResNet-50 slice encoder
- learnable slice positional embedding
- relative position enhanced attention (RoPE + relative position bias)
- PPEG module
- combined optimization with cross-entropy loss, focal loss, and prototype loss

### `gradcam.py`
This script generates Grad-CAM visualizations based on the trained model. It:
- loads a trained checkpoint
- reads prediction results from CSV
- computes slice-wise Grad-CAM heatmaps on the shared ResNet-50 encoder
- overlays heatmaps on the original slices and saves the visualization grids

`gradcam.py` supports two processing modes controlled by `PROCESS_MODE`: `"correct_only"` (default) for correctly predicted cases only, and `"all"` for all cases recorded in the prediction CSV.


## 3. Software environment
The code was developed in Python and depends mainly on the following packages:
- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- nibabel
- SimpleITK
- OpenCV (`cv2`)
- matplotlib
- seaborn
- tqdm
- easydict
- Jupyter Notebook
- `nystrom_attention`
- YOLOX-related modules (used in `preprocess.ipynb`)

GPU acceleration is recommended for model training and Grad-CAM generation, but CPU execution is possible for small-scale inspection.

## 4. Expected data organization
The training/inference script expects the ROI data to be organized as follows:

```text
project_root/
├── datasets/
│   ├── train/
│   │   ├── CAA/
│   │   ├── CH/
│   │   ├── IP-I/
│   │   ├── IP-II/
│   │   ├── IP-III/
│   │   ├── Normal/
│   │   └── SM/
│   └── test/
│       ├── CAA/
│       ├── CH/
│       ├── IP-I/
│       ├── IP-II/
│       ├── IP-III/
│       ├── Normal/
│       └── SM/
```

Each file corresponds to one inner ear ROI volume in NIfTI format (`.nii.gz`).

## 5. Recommended execution order

### Step 1. Preprocessing
Run `preprocess.ipynb` to:
- convert raw DICOM series to NIfTI volumes
- resample the volumes to a unified spacing
- localize the inner ear ROI using the YOLOX detector
- crop and save left/right ROI volumes

Before running the ROI localization cells, please update the local paths. The notebook example uses a temporal bone ROI physical size of `20 × 20 × 20 mm`. The YOLOX checkpoint used for ROI localization is not included in this supplementary package.



### Step 2. Dataset preparation
Organize the cropped ROI volumes into class-specific folders under `datasets/train/` and `datasets/test/`.

### Step 3. Model training or inference
Run `TransMIL_main.py`.


Important configurable variables include:
- `DATA_ROOT`
- `NUM_SLICES`
- `BATCH_SIZE`
- `EPOCHS`
- `SAVE_DIR`
- `MODE` (`"train"` or `"predict"`)
- `ONLY_TEST`

Typical usage:
- set `MODE = "train"` to train the model
- set `MODE = "predict"` to run inference using a saved checkpoint

### Step 4. Explainability visualization
Run `gradcam.py` after prediction files and checkpoints have been generated.

This script expects:
- a trained checkpoint such as `best_ACC.pth`
- a prediction CSV such as `test_predictions_ACC.csv`
- `PROCESS_MODE = "correct_only"` or `PROCESS_MODE = "all"`, depending on whether Grad-CAM should be generated for correctly predicted cases only or for all cases


## 6. Main outputs

### Outputs from `preprocess.ipynb`
- converted NIfTI volumes
- resampled NIfTI volumes
- cropped left/right inner ear ROI volumes

### Outputs from `TransMIL_main.py`
Depending on the mode, the script may generate:

- trained model checkpoints (e.g., `best_ACC.pth`)
- metrics CSV files
- prediction CSV files
- ROC curves
- top-K slice CSV files
- exported top-K slice images for test cases

### Outputs from `gradcam.py`
- Grad-CAM overlay figures saved under the specified output directory
- one visualization grid per processed case

## 7. Important implementation notes
1. The file paths in the scripts are placeholders and should be adapted to the local environment before execution.
2. `preprocess.ipynb` contains hard-coded example paths that should be updated before use.
3. `gradcam.py` imports `SliceTransMIL` directly from `TransMIL_main.py`, and the submitted filenames are fully consistent.
4. `gradcam.py` is designed for post hoc visualization and is not required for model training.
5. The code is intended for research verification and not for direct clinical deployment.


## 8. Notes for peer review
- No patient-identifiable data are included in this supplementary code package. Data availability is subject to institutional ethics and data security restrictions.
- The model code is provided as supplementary material for peer review and will be made publicly available on GitHub upon acceptance of the manuscript.
- If additional information is required to understand the scripts or reproduce specific outputs, it can be provided by the corresponding/lead contact upon request.

## 9. Contact
For questions regarding code structure, execution details, or output interpretation, please contact the corresponding/lead author.
