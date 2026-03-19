import numpy as np
import pandas as pd
import librosa as lb
from smart_stethoscope.params import *


# ===================================
# Core Audio Preprocessing (Production)
# Used by current live model + API
# ===================================

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


# ===================================
# Optional Audio Feature Extraction
# (for use in model exploration)
# ===================================

def extract_mfcc_feature_map(padded_audio: np.ndarray) -> np.ndarray:
    """
    Extract full MFCC feature map (time-frequency representation).

    Retains temporal structure for experimentation with CNNs
    or alternative architectures.

    Not currently used in production inference.
    """
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

def extract_mfcc_summary_features(df, audio_folder, n_mfcc=13):
    """
    Extract summary MFCC features for classical ML models.

    For each breathing cycle:
    - Compute MFCC matrix (n_mfcc x time_frames)
    - Collapse the time dimension using summary statistics
      (mean, std, skewness, max) per coefficient

    Returns:
        pd.DataFrame where each row corresponds to a breathing cycle
        and columns contain aggregated MFCC statistics.

    Output shape per sample:
        (n_mfcc * 4,)  -> suitable for tabular models
        such as Logistic Regression, Random Forest, etc.
    """
    mfcc_rows = []

    for cycle_filename in df["cycle_filename"]:
        file_path = Path(audio_folder) / f"{cycle_filename}.wav"

        signal, sample_rate = librosa.load(file_path, sr=None)

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=n_mfcc
        )

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_skew = skew(mfcc, axis=1)
        mfcc_max = np.max(mfcc, axis=1)

        combined = np.concatenate([mfcc_mean, mfcc_std, mfcc_skew, mfcc_max])

        mfcc_rows.append([cycle_filename] + list(combined))

    columns = ["cycle_filename"]

    for i in range(1, n_mfcc + 1):
        columns.append(f"mfcc_{i}_mean")
    for i in range(1, n_mfcc + 1):
        columns.append(f"mfcc_{i}_std")
    for i in range(1, n_mfcc + 1):
        columns.append(f"mfcc_{i}_skew")
    for i in range(1, n_mfcc + 1):
        columns.append(f"mfcc_{i}_max")

    return pd.DataFrame(mfcc_rows, columns=columns)
