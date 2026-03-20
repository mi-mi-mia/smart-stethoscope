from smart_stethoscope.ml_logic.data_loading import load_data
from smart_stethoscope.ml_logic.preprocessing import preprocess_tabular_data
from smart_stethoscope.ml_logic.audio_preprocessing import (
    preprocess_audio,
    audio_feature_extraction,
)
import numpy as np
import pandas as pd


def preprocessing():
    data = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, train_cycle_filenames, val_cycle_filenames, test_cycle_filenames = (
        preprocess_tabular_data(
            data, pipeline_save_path="models/post_split_pipeline.pkl"
        )
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, train_cycle_filenames, val_cycle_filenames, test_cycle_filenames


def train():
    pass


### This is for a CNN model (no tabular data yet)
def predict(audio: np.ndarray, original_sampling_rate: int, annotations: pd.DataFrame):
    padded_audio = preprocess_audio(audio, original_sampling_rate, annotations)
    breathing_cycle_features = audio_feature_extraction(padded_audio)
    # load model
    # for each brething cycle in breathing_cycle_features predict the class
    pass


if __name__ == "__main__":
    preprocessing()  # safety recommendation
