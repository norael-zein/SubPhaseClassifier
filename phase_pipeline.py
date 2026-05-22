"""
Hierarchical prediction pipeline for CT subphase classification.

Main model predicts AP/DP/NP/VP.
- If main predicts AP, EAP/LAP submodel refines AP to EAP or LAP.
- If main predicts VP, NEP/VP submodel refines VP to NEP or VP.
- If main predicts DP or NP, that label is returned directly.
"""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from feature_extractor import MISSING_VALUE, get_features

MODEL_DIR = Path(__file__).resolve().parent / "final_models"

def load_model_bundle(model_dir, name):
    """Load a trained model, its selected features, and its label encoder."""
    
    model = joblib.load(model_dir / f"final_model_{name}.pkl")
    feature_names = joblib.load(model_dir / f"features_{name}.pkl")
    label_encoder = joblib.load(model_dir / f"label_encoder_{name}.pkl")
    return model, feature_names, label_encoder

def prepare_X(features, feature_names):
    """Prepare the feature table so it matches the features used during training."""

    X = features.reindex(columns=feature_names, fill_value=MISSING_VALUE)
    X = X.replace([np.inf, -np.inf], MISSING_VALUE)
    return X.fillna(MISSING_VALUE)

def predict_with_bundle(model, feature_names, label_encoder, features):
    """Run prediction for one model and return the predicted label and probabilities."""

    X = prepare_X(features, feature_names)
    predicted_number = model.predict(X)
    predicted_label = label_encoder.inverse_transform(predicted_number)[0]
    probabilities = None
    if hasattr(model, "predict_proba"):
        proba_labels = label_encoder.inverse_transform(model.classes_)
        probabilities = pd.Series(model.predict_proba(X)[0], index=proba_labels).sort_values(ascending=False)
    return predicted_label, probabilities, X

def predict_phase(input_path, fast=False, totalseg_model_dir=None, model_dir=MODEL_DIR):
    """Predict the final CT phase using the main model and submodels when needed."""

    model_dir = Path(model_dir)
    main = load_model_bundle(model_dir, "main")
    eap_lap = load_model_bundle(model_dir, "EAP_LAP")
    nep = load_model_bundle(model_dir, "NEP")
    features = get_features(input_data=input_path, fast=fast, totalseg_model_dir=totalseg_model_dir)
    main_label, main_probabilities, X_main = predict_with_bundle(*main, features)
    final_label = main_label
    submodel_name = None
    sub_probabilities = None
    X_sub = None

    if main_label == "AP":
        final_label, sub_probabilities, X_sub = predict_with_bundle(*eap_lap, features)
        submodel_name = "EAP_LAP"
    elif main_label == "VP":
        final_label, sub_probabilities, X_sub = predict_with_bundle(*nep, features)
        submodel_name = "NEP"
    return {"final_phase": final_label,"main_phase": main_label,"submodel": submodel_name,"main_probabilities": main_probabilities,"sub_probabilities": sub_probabilities,"features_all": features,"X_main": X_main,"X_sub": X_sub}

def print_probabilities(title, probabilities):
    """Print predicted class probabilities."""
    if probabilities is None:
        return
    print(f"\n{title}:")
    for phase_name, value in probabilities.items():
        print(f"{phase_name}: {value:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to a .nii.gz CT image or DICOM folder")
    parser.add_argument("--fast", action="store_true", help="Use fast TotalSegmentator")
    parser.add_argument("--totalseg-model-dir", default=None, help="Optional path to a local TotalSegmentator model folder")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Folder containing final_model_*.pkl, features_*.pkl, label_encoder_*.pkl")
    parser.add_argument("--save-features", default=None, help="Optional CSV path for all extracted features")
    args = parser.parse_args()
    result = predict_phase(input_path=args.input_path,fast=args.fast,totalseg_model_dir=args.totalseg_model_dir,model_dir=args.model_dir)

    print(f"\nMain model prediction: {result['main_phase']}")
    print(f"Final predicted phase: {result['final_phase']}")
    if result["submodel"] is not None:
        print(f"Submodel used: {result['submodel']}")
    print_probabilities("Main probabilities", result["main_probabilities"])
    print_probabilities("Submodel probabilities", result["sub_probabilities"])
    if args.save_features is not None:
        result["features_all"].to_csv(args.save_features, index=False)
        print(f"\nSaved extracted features to: {args.save_features}")

if __name__ == "__main__":
    main()
