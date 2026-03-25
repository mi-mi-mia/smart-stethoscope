import numpy as np
import pandas as pd
import librosa as lb
from smart_stethoscope.params import *
from pathlib import Path
from scipy.stats import skew, kurtosis


# ===================================
# Core Audio Preprocessing (Production)
# Used by current live model + API
# ===================================

MU = 255
LOG_MU = np.log1p(MU)


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


def extract_audio_segments(audio, start, end, diagnosis):
    audio = cut_audio_data(audio, start, end, sr=TARGET_SAMPLING_RATE)

    frames_per_segment = SEGMENT_LENGTH * TARGET_SAMPLING_RATE
    step_size = (
        frames_per_segment
        if diagnosis in {"COPD", "Unknown"}
        else STEP_LENGTH * TARGET_SAMPLING_RATE
    )

    n_segments = (len(audio) - frames_per_segment) // step_size + 1
    if n_segments <= 0:
        return []

    shape = (n_segments, frames_per_segment)
    strides = (audio.strides[0] * step_size, audio.strides[0])

    segments = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

    return segments


def compress_audio(audio):
    return np.sign(audio) * np.log1p(MU * np.abs(audio)) / LOG_MU


def extract_audio_features(audio):
    return (
        extract_numerical_audio_features(audio),
        extract_mel_spectrogram(audio),
    )


def extract_numerical_audio_features(audio):
    features = {}

    # Precompute STFT magnitude
    S = np.abs(lb.stft(audio))

    # TEMPORAL
    rms = lb.feature.rms(S=S)
    zcr = lb.feature.zero_crossing_rate(y=audio)

    features["rms_mean"] = float(rms.mean())
    features["rms_std"] = float(rms.std())
    features["zcr_mean"] = float(zcr.mean())

    # SPECTRAL
    centroid = lb.feature.spectral_centroid(S=S, sr=TARGET_SAMPLING_RATE)
    bandwidth = lb.feature.spectral_bandwidth(S=S, sr=TARGET_SAMPLING_RATE)
    rolloff = lb.feature.spectral_rolloff(S=S, sr=TARGET_SAMPLING_RATE)
    flatness = lb.feature.spectral_flatness(S=S)

    features["centroid_mean"] = float(centroid.mean())
    features["centroid_std"] = float(centroid.std())

    features["flatness_mean"] = float(flatness.mean())
    features["flatness_std"] = float(flatness.std())

    features["rolloff_mean"] = float(rolloff.mean())
    features["bandwidth_mean"] = float(bandwidth.mean())

    # Flux (onset strength)
    flux = lb.onset.onset_strength(S=S, sr=TARGET_SAMPLING_RATE)
    features["flux_mean"] = float(flux.mean())

    # SHAPE
    features["skewness_mean"] = float(skew(centroid, axis=1)[0])
    features["kurtosis_mean"] = float(kurtosis(centroid, axis=1)[0])

    # MFCC (reuses S implicitly)
    mfccs = lb.feature.mfcc(S=lb.power_to_db(S**2), n_mfcc=16)

    mfcc_mean = mfccs.mean(axis=1)
    mfcc_std = mfccs.std(axis=1)
    for i in range(1, 16):
        features[f"mfcc_{i}_mean"] = float(mfcc_mean[i])
        features[f"mfcc_{i}_std"] = float(mfcc_std[i])

    return features


def extract_mel_spectrogram(
    breathing_cycle: np.ndarray,
    sample_rate: int = TARGET_SAMPLING_RATE,
    n_mels: int = 128,
    max_time_steps: int = 200,
) -> np.ndarray:
    """
    Convert one breathing-cycle waveform into the mel spectrogram format
    used by the production CNN model.

    Steps
    -----
    - Compute mel spectrogram from waveform
    - Convert power spectrogram to dB scale
    - Pad or crop the time axis to a fixed width
    - Add channel dimension for CNN input

    Parameters
    ----------
    breathing_cycle : np.ndarray
        One breathing-cycle waveform.
    sample_rate : int, default=TARGET_SAMPLING_RATE
        Sampling rate of the waveform.
    n_mels : int, default=64
        Number of mel frequency bins.
    max_time_steps : int, default=200
        Fixed width for the spectrogram. Shorter spectrograms are padded
        with zeros on the right; longer ones are cropped on the right.

    Returns
    -------
    np.ndarray
        Mel spectrogram of shape (n_mels, max_time_steps, 1),
        ready for CNN input.
    """
    mel_spec = lb.feature.melspectrogram(
        y=breathing_cycle, sr=sample_rate, n_mels=n_mels
    )

    mel_spec_db = lb.power_to_db(mel_spec, ref=np.max)

    current_time_steps = mel_spec_db.shape[1]

    if current_time_steps < max_time_steps:
        pad_width = max_time_steps - current_time_steps
        mel_spec_db = np.pad(
            mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode="constant"
        )
    elif current_time_steps > max_time_steps:
        mel_spec_db = mel_spec_db[:, :max_time_steps]

    mel_spec_db = mel_spec_db[..., np.newaxis]

    return mel_spec_db.astype(np.float32)


def audio_preprocessing(audio, sampling_rate, start, end):
    if sampling_rate != TARGET_SAMPLING_RATE:
        audio = lb.resample(
            y=audio, orig_sr=sampling_rate, target_sr=TARGET_SAMPLING_RATE
        )
    audio_segments = extract_audio_segments(audio, start, end, diagnosis="Unknown")

    if len(audio_segments) == 0:
        return pd.DataFrame(), np.empty((0,), dtype=np.float32)

    n = len(audio_segments)
    features_list = [None] * n
    mel_list = [None] * n

    for i, segment in enumerate(audio_segments):
        segment_compressed = compress_audio(segment)
        features, mel = extract_audio_features(segment_compressed)

        features_list[i] = features
        mel_list[i] = mel

    features_df = pd.DataFrame(features_list)
    mel_spectograms = np.stack(mel_list).astype(np.float32)

    return features_df, mel_spectograms
