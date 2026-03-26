from smart_stethoscope.ml_logic.data_loading import load_and_preprocess_raw_audio_data
from smart_stethoscope.ml_logic.preprocessing import audio_preprocessing
from smart_stethoscope.ml_logic.model import train_final_hybrid_models, predict_hybrid
from smart_stethoscope.params import CLASS_NAMES
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def preprocessing():
    features_df, mel_spec = load_and_preprocess_raw_audio_data()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(features_df["diagnosis"])

    groups = features_df["patient_id"]
    X = features_df.drop(columns=["patient_id", "diagnosis"], errors="ignore")

    return X, mel_spec, y, groups


def train(n_splits=3, random_state=42):
    """
    Run the full training pipeline for the hybrid model.

    Returns
    -------
    dict
        {
            "xgb_model": trained XGB model,
            "cnn_model": trained CNN model,
            "train_idx": train indices,
            "val_idx": validation indices,
            "X": tabular features,
            "mel_spec": CNN input array,
            "y": encoded labels,
            "groups": patient groups
        }
    """
    X, mel_spec, y, groups = preprocessing()

    results = train_final_hybrid_models(
        X=X,
        y=y,
        mel_spec=mel_spec,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
    )

    return {
        "xgb_model": results["xgb_model"],
        "cnn_model": results["cnn_model"],
        "train_idx": results["train_idx"],
        "val_idx": results["val_idx"],
        "X": X,
        "mel_spec": mel_spec,
        "y": y,
        "groups": groups,
    }


def preprocess_for_prediction(audio, sampling_rate, start, end):
    return audio_preprocessing(audio, sampling_rate, start, end)


def predict(xgb_model, cnn_model, xgb_features, cnn_features):
    return predict_hybrid(xgb_model, cnn_model, xgb_features, cnn_features, class_names=CLASS_NAMES)


if __name__ == "__main__":
    results = train()

    print("Training complete")
    print("Train size:", len(results["train_idx"]))
    print("Val size:", len(results["val_idx"]))
