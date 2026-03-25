from smart_stethoscope.ml_logic.data_loading import load_audio_data
from smart_stethoscope.params import CLASSES_TO_KEEP
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def preprocessing():
    features_df, mel_spectograms = load_audio_data()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(features_df["diagnosis"])

    groups = features_df["patient_id"]
    X = features_df.drop(columns=["patient_id", "diagnosis"], errors="ignore")

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
