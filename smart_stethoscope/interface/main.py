from smart_stethoscope.ml_logic.data_loading import load_audio_data
from smart_stethoscope.params import CLASSES_TO_KEEP
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def preprocessing():
    features_df, mel_spectograms = load_audio_data()

    # encode - sequentially (0, 1, 2...)
    label_encoder = LabelEncoder()
    features_df["target"] = label_encoder.fit_transform(features_df["diagnosis"])

    # define X (features), y (target) and groups
    cols_to_drop = ["patient_id", "diagnosis", "target"]
    X = features_df.drop(columns=[c for c in cols_to_drop if c in features_df.columns])
    y = features_df["target"]
    groups = features_df["patient_id"]
    return X, mel_spectograms, y, groups


def train():
    pass


def preprocess_for_prediction(audio, sampling_rate, start, end):
    return features_df, mel_spectograms


### This is for a CNN model (no tabular data yet)
def predict(audio: np.ndarray, original_sampling_rate: int, annotations: pd.DataFrame):
    padded_audio = preprocess_audio(audio, original_sampling_rate, annotations)
    # audio feature extraction
    # load model
    # for each brething cycle in breathing_cycle_features predict the class
    pass


if __name__ == "__main__":
    preprocessing()  # safety recommendation
