"""
Feature extraction for CT subphase classification.
"""
import os
import tempfile
from contextlib import contextmanager
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.python_api import totalsegmentator

MISSING_VALUE = -9999
ORGANS = ["aorta","portal_vein_and_splenic_vein","urinary_bladder","kidney_right","kidney_left","spleen","liver"]
FINAL_LABELS = ["aorta","kidneys","liver","portal_vein_and_splenic_vein","spleen","urinary_bladder"]
DIFF_FEATURE_TYPES = ["Mean","Median","Variance","Standard Deviation","10th Percentile","50th Percentile","90th Percentile","InterquartileRange (IQR)","RootMeanSquared (RMS)","Skewness","Kurtosis","Entropy","Uniformity","GLCM_Contrast","GLCM_Correlation","GLCM_JointEntropy","GLSZM_ZoneVariance","GLSZM_ZoneEntropy","GLDM_DependenceEntropy","GLRLM_RunEntropy"]
_EXTRACTOR = None

@contextmanager
def totalseg_model_dir(model_dir=None):
    """Temporarily set the TotalSegmentator model directory."""

    old_value = os.environ.get("TOTALSEG_HOME_DIR")
    if model_dir is None:
        os.environ.pop("TOTALSEG_HOME_DIR", None)
    else:
        os.environ["TOTALSEG_HOME_DIR"] = str(model_dir)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("TOTALSEG_HOME_DIR", None)
        else:
            os.environ["TOTALSEG_HOME_DIR"] = old_value

def run_totalsegmentator(input_data, task="total", roi_subset=None, fast=False, model_dir=None):
    """Run TotalSegmentator and return the segmentation result."""

    print(f"Running TotalSegmentator for task '{task}'...")
    with totalseg_model_dir(model_dir):
        return totalsegmentator(input=input_data,output=None,task=task,roi_subset=roi_subset,ml=True,skip_saving=True,fast=fast)

def get_extractor():
    """Create or reuse the PyRadiomics feature extractor."""

    global _EXTRACTOR
    if _EXTRACTOR is None:
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeaturesByName(
            firstorder=["Mean","Median","Variance","10Percentile","90Percentile","InterquartileRange","RootMeanSquared","Skewness","Kurtosis","Entropy","Uniformity"],
            glcm=["Contrast", "Correlation", "JointEntropy"],
            glszm=["ZoneVariance", "ZoneEntropy"],
            gldm=["DependenceEntropy"],
            glrlm=["RunEntropy"],
        )
        extractor.disableAllImageTypes()
        extractor.enableImageTypeByName("Original")
        _EXTRACTOR = extractor
    return _EXTRACTOR

def read_ct_image(input_data):
    """Read a CT image from a SimpleITK image, DICOM folder, or image file."""

    if isinstance(input_data, sitk.Image):
        return input_data
    if os.path.isdir(str(input_data)):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(input_data))
        if len(dicom_names) == 0:
            raise ValueError(f"No DICOM files found in folder: {input_data}")
        reader.SetFileNames(dicom_names)
        return reader.Execute()
    return sitk.ReadImage(str(input_data))

def clean_value(value):
    """Convert invalid numeric values to the missing value marker."""

    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return MISSING_VALUE
    return value

def nibabel_mask_to_sitk(mask_np, reference_sitk):
    """Convert a NumPy mask from TotalSegmentator to a SimpleITK mask."""

    mask_sitk = sitk.GetImageFromArray(mask_np.transpose(2, 1, 0).astype(np.uint8))
    mask_sitk.CopyInformation(reference_sitk)
    return mask_sitk

def mask_is_empty(mask):
    """Check if a segmentation mask contains any voxels."""

    return np.sum(sitk.GetArrayFromImage(mask) > 0) == 0

def selected_features(raw_features, label_name):
    """Select and rename the radiomic features."""

    variance = clean_value(raw_features["original_firstorder_Variance"])
    return {
        f"Mean_{label_name}": clean_value(raw_features["original_firstorder_Mean"]),
        f"Median_{label_name}": clean_value(raw_features["original_firstorder_Median"]),
        f"Variance_{label_name}": variance,
        f"Standard Deviation_{label_name}": clean_value(np.sqrt(max(variance, 0))),
        f"10th Percentile_{label_name}": clean_value(raw_features["original_firstorder_10Percentile"]),
        f"50th Percentile_{label_name}": clean_value(raw_features["original_firstorder_Median"]),
        f"90th Percentile_{label_name}": clean_value(raw_features["original_firstorder_90Percentile"]),
        f"InterquartileRange (IQR)_{label_name}": clean_value(raw_features["original_firstorder_InterquartileRange"]),
        f"RootMeanSquared (RMS)_{label_name}": clean_value(raw_features["original_firstorder_RootMeanSquared"]),
        f"Skewness_{label_name}": clean_value(raw_features["original_firstorder_Skewness"]),
        f"Kurtosis_{label_name}": clean_value(raw_features["original_firstorder_Kurtosis"]),
        f"Entropy_{label_name}": clean_value(raw_features["original_firstorder_Entropy"]),
        f"Uniformity_{label_name}": clean_value(raw_features["original_firstorder_Uniformity"]),
        f"GLCM_Contrast_{label_name}": clean_value(raw_features["original_glcm_Contrast"]),
        f"GLCM_Correlation_{label_name}": clean_value(raw_features["original_glcm_Correlation"]),
        f"GLCM_JointEntropy_{label_name}": clean_value(raw_features["original_glcm_JointEntropy"]),
        f"GLSZM_ZoneVariance_{label_name}": clean_value(raw_features["original_glszm_ZoneVariance"]),
        f"GLSZM_ZoneEntropy_{label_name}": clean_value(raw_features["original_glszm_ZoneEntropy"]),
        f"GLDM_DependenceEntropy_{label_name}": clean_value(raw_features["original_gldm_DependenceEntropy"]),
        f"GLRLM_RunEntropy_{label_name}": clean_value(raw_features["original_glrlm_RunEntropy"]),
    }

def extract_features_for_mask(image, mask, label_name):
    """Extract radiomic features from one segmentation mask."""

    if mask_is_empty(mask):
        print(f"Warning: empty mask for {label_name}")
        return {}
    extractor = get_extractor()
    try:
        raw_features = extractor.execute(image, mask)
        return selected_features(raw_features, label_name)
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"Could not extract features for {label_name}: {error}")
        return {}

def combine_feature_sets(feature_sets, old_suffixes, new_suffix):
    """Combine features from several masks into one shared region."""

    combined = {}
    all_keys = set().union(*(features.keys() for features in feature_sets))
    for key in all_keys:
        values = []
        output_key = key
        for old_suffix in old_suffixes:
            output_key = output_key.replace(f"_{old_suffix}", f"_{new_suffix}")
        for features in feature_sets:
            if key in features and features[key] != MISSING_VALUE:
                values.append(float(features[key]))
        if values:
            combined[output_key] = float(np.mean(values))
    return combined

def extract_labels_from_segmentation(image, seg_img, task, labels):
    """Extract features for selected labels from a TotalSegmentator result."""

    seg_data = np.asanyarray(seg_img.dataobj)
    label_to_id = {label_name: label_id for label_id, label_name in class_map[task].items()}
    output = {}
    for label in labels:
        if label not in label_to_id:
            print(f"Warning: {label} is not in TotalSegmentator class_map for {task}")
            continue
        mask_np = seg_data == label_to_id[label]
        if mask_np.sum() == 0:
            print(f"Warning: empty mask for {label}")
            continue
        mask = nibabel_mask_to_sitk(mask_np, image)
        output[label] = extract_features_for_mask(image, mask, label)
    return output

def get_feature_value(features, feature_name):
    """Get one feature value and handle missing values."""

    value = features.get(feature_name, MISSING_VALUE)
    if value == MISSING_VALUE:
        return MISSING_VALUE
    return clean_value(value)

def add_difference_features(features):
    """Add pairwise difference features between the final anatomical regions."""

    for feature_type in DIFF_FEATURE_TYPES:
        for label_1 in FINAL_LABELS:
            for label_2 in FINAL_LABELS:
                if label_1 == label_2:
                    continue
                output_name = f"{feature_type}_diff_{label_1}_{label_2}"
                value_1 = get_feature_value(features, f"{feature_type}_{label_1}")
                value_2 = get_feature_value(features, f"{feature_type}_{label_2}")
                if value_1 == MISSING_VALUE or value_2 == MISSING_VALUE:
                    features[output_name] = MISSING_VALUE
                else:
                    features[output_name] = value_1 - value_2
    return features

def get_features(input_data, fast=False, totalseg_model_dir=None):
    """Run segmentation and feature extraction for one CT scan."""

    image = read_ct_image(input_data)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.nii.gz")
        sitk.WriteImage(image, input_path)
        print("Step 1/2: Segmenting organs...")
        organ_seg = run_totalsegmentator(input_path, task="total", roi_subset=ORGANS, fast=fast,model_dir=totalseg_model_dir)
        print("Step 2/2: Extracting radiomic features...")
        organ_features = extract_labels_from_segmentation(image, organ_seg, "total", ORGANS)

    features = {}
    for values in organ_features.values():
        features.update(values)
    if "kidney_right" in organ_features and "kidney_left" in organ_features:
        features.update(combine_feature_sets([organ_features["kidney_right"], organ_features["kidney_left"]],["kidney_right", "kidney_left"],"kidneys"))
    features = add_difference_features(features)
    return pd.DataFrame([features]).replace([np.inf, -np.inf], MISSING_VALUE).fillna(MISSING_VALUE)
