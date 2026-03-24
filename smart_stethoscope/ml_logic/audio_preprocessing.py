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
    if diagnosis == "COPD" or diagnosis == "Unknown":
        step_size = frames_per_segment
    else:
        step_size = STEP_LENGTH * TARGET_SAMPLING_RATE

    segments = []
    for start_idx in range(0, len(audio) - frames_per_segment + 1, step_size):
        end_idx = start_idx + frames_per_segment
        segment = audio[start_idx:end_idx]
        segments.append(segment)
    return segments


def compress_audio(audio):
    mu = 255
    return np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)


def extract_audio_features(audio):
    features = extract_numerical_audio_features(audio)
    mel_spectograms = extract_mel_spectrogram(audio)
    return features, mel_spectograms


def extract_numerical_audio_features(audio):
    features = {}

    # TEMPORAL
    rms = lb.feature.rms(y=audio)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))  # sound explosion

    zcr = lb.feature.zero_crossing_rate(y=audio)
    features["zcr_mean"] = float(np.mean(zcr))

    # SPECTRAL
    centroid = lb.feature.spectral_centroid(
        y=audio, sr=TARGET_SAMPLING_RATE
    )  # .flatten() # gemini told me to do it. centroids are 2D and we need it 1D apparently
    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_std"] = float(np.std(centroid))  # tone change

    # SHAPE STATISTICS
    flatness = lb.feature.spectral_flatness(y=audio)
    features["flatness_mean"] = float(np.mean(flatness))
    features["flatness_std"] = float(np.std(flatness))  # constant noise vs intermitent

    # OTHERS
    features["rolloff_mean"] = float(
        np.mean(lb.feature.spectral_rolloff(y=audio, sr=TARGET_SAMPLING_RATE))
    )
    features["flux_mean"] = float(
        np.mean(lb.onset.onset_strength(y=audio, sr=TARGET_SAMPLING_RATE))
    )
    features["bandwidth_mean"] = float(
        np.mean(lb.feature.spectral_bandwidth(y=audio, sr=TARGET_SAMPLING_RATE))
    )

    # FORM NEW! - depend on centroid
    features["skewness_mean"] = float(skew(centroid, axis=1)[0])
    features["kurtosis_mean"] = float(kurtosis(centroid, axis=1)[0])

    # MFCC (16, ignore 0)
    mfccs = lb.feature.mfcc(y=audio, sr=TARGET_SAMPLING_RATE, n_mfcc=16)
    for i in range(1, 16):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

    return features


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


def preprocess_audio(
    audio: np.ndarray, original_sampling_rate: int, annotations: pd.DataFrame
) -> np.ndarray:
    """
    Resample a raw respiratory recording, cut it into breathing cycles using
    annotation start/end times, and pad or trim each cycle to a fixed length.

    Returns
    -------
    np.ndarray
    Array of shape (num_cycles, fixed_num_samples) containing one padded
    waveform per breathing cycle.
    """
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


def build_mel_spectrogram_dataset(
    padded_audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLING_RATE,
    n_mels: int = 64,
    max_time_steps: int = 200,
) -> np.ndarray:
    """
    Convert an array of padded breathing cycles into a CNN-ready batch of
    mel spectrograms.

    Parameters
    ----------
    padded_audio : np.ndarray
        Array of shape (num_cycles, num_samples), where each row is one
        breathing-cycle waveform.
    sample_rate : int, default=TARGET_SAMPLING_RATE
        Sampling rate of the waveforms.
    n_mels : int, default=64
        Number of mel frequency bins.
    max_time_steps : int, default=200
        Fixed width for each spectrogram.

    Returns
    -------
    np.ndarray
        Array of shape (num_cycles, n_mels, max_time_steps, 1)
        suitable for CNN prediction.
    """
    mel_specs = []

    for breathing_cycle in padded_audio:
        mel = extract_mel_spectrogram(
            breathing_cycle=breathing_cycle,
            sample_rate=sample_rate,
            n_mels=n_mels,
            max_time_steps=max_time_steps,
        )
        mel_specs.append(mel)

    return np.stack(mel_specs).astype(np.float32)


# ===================================
# Optional Audio Feature Extraction
# (for use in model exploration)
# ===================================


def extract_mfcc_feature_map(padded_audio: np.ndarray) -> np.ndarray:
    """
    Extract full MFCC matrices from padded breathing cycles.
    Each breathing cycle is converted into an MFCC array of shape
    (n_mfcc, time_frames), preserving temporal structure.

    Returns
    -------
    np.ndarray
        Array of shape (num_cycles, n_mfcc, time_frames), suitable for
        experimentation with CNNs or other deep learning models.

    Notes
    -----
    This function is for model exploration and is not currently used in
    production inference.
    """
    n_mfcc = 13
    mfccs = []

    for breathing_cycle in padded_audio:
        mfcc = lb.feature.mfcc(
            y=breathing_cycle, sr=TARGET_SAMPLING_RATE, n_mfcc=n_mfcc
        )
        mfccs.append(mfcc)

    return np.stack(mfccs)


def extract_mfcc_summary_features(df, audio_folder, n_mfcc=13):
    """
    Extract aggregated MFCC summary features from pre-cut breathing-cycle .wav files.

    For each cycle file:
    - load waveform
    - compute MFCC matrix
    - summarise each coefficient across time using mean, std, skewness, and max

    Returns
    -------
    pd.DataFrame
        One row per cycle_filename with statistical MFCC features suitable for
        classical tabular models such as Logistic Regression.

    Notes
    -----
    This function is intended for model exploration and baseline modelling.
    It is not currently used in production inference.
    """
    mfcc_rows = []

    for cycle_filename in df["cycle_filename"]:
        file_path = Path(audio_folder) / f"{cycle_filename}.wav"

        signal, sample_rate = lb.load(file_path, sr=None)

        mfcc = lb.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)

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
