import numpy as np
import pandas as pd
import librosa as lb
from smart_stethoscope.params import *


def cut_audio_data(raw_data, start, end, sr=22050):
    """
    Slices a numpy array of audio data using start and end timestamps.

    Parameters
    ----------
    raw_data : np.array
        Numpy array of audio sample
    start : float
        Start time in seconds
    end : float
        End time in seconds
    sr : int
        Sampling rate (default 22050)

    Returns
    -------
    np.array : sliced audio data
    """
    max_ind = len(raw_data)
    start_ind = min(int(start * sr), max_ind)
    end_ind = min(int(end * sr), max_ind)
    return raw_data[start_ind:end_ind]


def pad_audio(breathing_cycle: np.ndarray):
    padding_length = int(AUDIO_LENGTH * TARGET_SAMPLING_RATE)
    if len(breathing_cycle) < padding_length:
        # Pad with zeros
        pad_width = padding_length - len(breathing_cycle)
        padded_audio = np.pad(breathing_cycle, (0, pad_width), mode="constant")
    else:
        # Trim
        padded_audio = breathing_cycle[:padding_length]
    return padded_audio


def audio_feature_extraction(padded_audio: np.ndarray) -> np.ndarray:
    # TODO: Depending on the model, extract the feature we want.
    # don't hard code n_mfcc here. make parameter as same like in training.
    # We should use the same feature extraction function for training and practicing!!!
    n_mfcc = 13
    mfccs = []

    for breathing_cycle in padded_audio:
        mfcc = lb.feature.mfcc(
            y=breathing_cycle, sr=TARGET_SAMPLING_RATE, n_mfcc=n_mfcc
        )
        mfccs.append(mfcc)

    return np.stack(mfccs)


def preprocess_audio(
    audio: np.ndarray, original_sampling_rate: int, annotations: pd.DataFrame
) -> np.ndarray:
    # Resample:
    audio = lb.resample(
        audio, orig_sr=original_sampling_rate, target_sr=TARGET_SAMPLING_RATE
    )
    padded_audios = []
    for row in annotations.itertuples(index=False):
        breathing_cycle = cut_audio_data(
            audio, row.start, row.end, TARGET_SAMPLING_RATE
        )
        padded_audio = pad_audio(breathing_cycle=breathing_cycle)
        padded_audios.append(padded_audio)
    padded_audios_array = np.stack(padded_audios).astype(np.float32)

    return padded_audios_array
