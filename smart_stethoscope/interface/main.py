from smart_stethoscope.ml_logic.data_loading import load_audio_data
from smart_stethoscope.ml_logic.preprocessing import preprocess_tabular_data
from smart_stethoscope.ml_logic.audio_preprocessing import preprocess_audio
import numpy as np
import pandas as pd


def preprocessing():
    features_df, mel_spectograms_array = load_audio_data()
    # TrainTestVal Split here or in model?
    return features_df, mel_spectograms_array


def train():
    pass


### This is for a CNN model (no tabular data yet)
def predict(audio: np.ndarray, original_sampling_rate: int, annotations: pd.DataFrame):
    padded_audio = preprocess_audio(audio, original_sampling_rate, annotations)
    # audio feature extraction
    # load model
    # for each brething cycle in breathing_cycle_features predict the class
    pass


if __name__ == "__main__":
    preprocessing()  # safety recommendation
