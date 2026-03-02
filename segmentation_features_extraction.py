"""
Extract features after segmenting out organs with TotalSegmentator.

"""
import os
import SimpleITK as sitk
import six
from radiomics import featureextractor, getTestCase
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator


def segment_selected_organs(input_data, output_path, organs, task="total", fast=False):
    """
    Segment out selected organs with TotalSegmentator.

    Parameters:
    input_data : NIfTI-file or DICOM files
    output_path : folder where the segmented output will be saved in NIfTI-format
    organs : list[str], list with organs
    task : Task-name, Default is "total". 
    fast: If True, the model will be run in fast mode, which is faster but less accurate. Default is False.
    See https://github.com/wasserth/TotalSegmentator
    """

    if organs is None or len(organs) == 0:
        raise ValueError("You need to input at least one organ.")

    result = totalsegmentator(input_data, output_path, task=task, roi_subset=organs)

    return result


def extract_features(image_path, mask_path):
    """
    Load an image and its corresponding mask, then extract radiomic features.
    """

    #Create a feature extractor instance
    feature_extractor = featureextractor.RadiomicsFeatureExtractor()

    #Read the image and mask and perform feature extraction
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    extracted_features = feature_extractor.execute(image, mask)

    return extracted_features

def calculate_feature_differences(organ_features):
    """
    Calculates pairwise differences for each statistical feature between organs.

    Args:
    organ_features (dict): Dictionary where each key is an organ name and each value is a dictionary of that organ’s features.

    Returns:
    dict: Dictionary containing the difference for each feature between all organ pairs.
    """
    differences = {}
    organ_names = list(organ_features.keys())
    
    # Get all feature names from the first organ (assuming same features for all organs)
    if not organ_names or not organ_features[organ_names[0]]:
        return differences
    
    feature_names = list(organ_features[organ_names[0]].keys())
    
    #For each feature, calculate differences between organs
    for feature_name in feature_names:
        feature_diff_key = f"{feature_name}_differences"
        differences[feature_diff_key] = {}
        
        #Calculate pairwise differences between organs
        for i in range(len(organ_names)):
            for j in range(i + 1, len(organ_names)):
                organ_i = organ_names[i]
                organ_j = organ_names[j]
                
                feature_i = organ_features[organ_i].get(feature_name)
                feature_j = organ_features[organ_j].get(feature_name)
                
                #Only calculate if both features exist and are numeric
                if feature_i is not None and feature_j is not None:
                    try:
                        diff_value = float(feature_i) - float(feature_j)
                        diff_key = f"{organ_i}_vs_{organ_j}"
                        differences[feature_diff_key][diff_key] = diff_value
                    except (ValueError, TypeError):
                        #Skip non-numeric features
                        pass
    
    return differences

