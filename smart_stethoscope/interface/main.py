from smart_stethoscope.ml_logic.data_loading import load_and_preprocess_raw_audio_data
from smart_stethoscope.ml_logic.preprocessing import audio_preprocessing
from smart_stethoscope.ml_logic.model import predict_hybrid
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


def predict(xgb_model, cnn_model, xgb_df, cnn_array):
    return predict_hybrid(xgb_model, cnn_model, xgb_df, cnn_array)


if __name__ == "__main__":
    preprocessing()  # safety recommendation
