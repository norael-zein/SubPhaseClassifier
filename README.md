# Total CT Phase Classifier

This project provides a pipeline for automatic CT phase and subphase classification using organ segmentation, radiomics feature extraction, and trained machine learning models.

![Pipeline overview](<Pipeline Arbete 2.png>)

The pipeline first predicts one of the main CT contrast phases:

- **NP**: Non-Contrast Phase
- **AP**: Arterial Phase
- **VP**: Portal-Venous Phase
- **DP**: Delayed Phase

If the scan is predicted as **AP**, an additional submodel is used to classify the scan as:

- **EAP**: Early Arterial Phase
- **LAP**: Late Arterial Phase

If the scan is predicted as **VP**, another submodel is used to classify the scan as:

- **VP**: Portal-Venous Phase
- **NEP**: Nephrographic Phase

Scans predicted as **NP** or **DP** are returned directly without further subphase classification.

---

## Pipeline Overview

The pipeline consists of the following steps:

1. Read a CT image from a DICOM folder or a NIfTI file.
2. Segment selected anatomical structures using TotalSegmentator.
3. Extract radiomic features from the segmentation masks using PyRadiomics.
4. Prepare the extracted features so they match the selected features used during training.
5. Predict the main CT phase using the main classification model.
6. Apply a subphase classifier if the main phase is predicted as AP or VP.

The following anatomical structures are used for feature extraction:

- Aorta
- Portal vein and splenic vein
- Urinary bladder
- Kidneys
- Spleen
- Liver

---

## Project Files

- `feature_extractor.py`  
  Runs TotalSegmentator and extracts radiomic features from the selected anatomical structures.

- `phase_pipeline.py`  
  Runs the full hierarchical prediction pipeline and returns the final CT phase prediction.

- `requirements.txt`  
  Contains the Python packages needed to run the project.

- `final_models/`  
  Folder containing the trained models, selected feature lists, and label encoders.

---

## Requirements

This project requires Python 3.10 or 3.11.

Install the required packages using:

```bash
pip install -r requirements.txt
```

The main packages used in this project are:

- TotalSegmentator
- PyRadiomics
- SimpleITK
- pandas
- scikit-learn
- XGBoost
- joblib

---

## How to Run

Run the pipeline from the command line:

```bash
python phase_pipeline.py "input_path"
```

The `input_path` can be either:

- a DICOM folder
- a NIfTI file, for example `.nii.gz`

Example:

```bash
python phase_pipeline.py "path/to/ct_scan.nii.gz"
```

or:

```bash
python phase_pipeline.py "path/to/dicom_folder"
```

---

## Optional Arguments

Use the fast TotalSegmentator option:

```bash
python phase_pipeline.py "input_path" --fast
```

Use a local TotalSegmentator model directory:

```bash
python phase_pipeline.py "input_path" --totalseg-model-dir "path/to/totalseg_models"
```

Save the extracted features to a CSV file:

```bash
python phase_pipeline.py "input_path" --save-features extracted_features.csv
```

Use a custom model directory:

```bash
python phase_pipeline.py "input_path" --model-dir "path/to/final_models"
```

---

## Output

The pipeline prints:

- the main phase prediction
- the final predicted phase
- the submodel used, if applicable
- class probabilities when available

If `--save-features` is used, the extracted features are also saved as a CSV file.

---

## References

- TotalSegmentator: https://github.com/wasserth/TotalSegmentator
- PyRadiomics: https://pyradiomics.readthedocs.io/
