from smart_stethoscope.ml_logic.data_loading import load_and_preprocess_raw_audio_data
from smart_stethoscope.ml_logic.preprocessing import audio_preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def preprocessing():
    features_df, mel_spectograms = load_and_preprocess_raw_audio_data()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(features_df["diagnosis"])

    groups = features_df["patient_id"]
    X = features_df.drop(columns=["patient_id", "diagnosis"], errors="ignore")

    return X, mel_spectograms, y, groups


def train():
    pass


def preprocess_for_prediction(audio, sampling_rate, start, end):
    return audio_preprocessing(audio, sampling_rate, start, end)


### This is for a CNN model (no tabular data yet)
def predict(audio: np.ndarray, original_sampling_rate: int, annotations: pd.DataFrame):
    pass


if __name__ == "__main__":
    preprocessing()  # safety recommendation
