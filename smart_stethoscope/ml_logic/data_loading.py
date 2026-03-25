from colorama import Fore, Style
import pandas as pd
import numpy as np
import librosa as lb
from joblib import Parallel, delayed
from smart_stethoscope.params import *
from smart_stethoscope.ml_logic.preprocessing import (
    extract_audio_segments,
    compress_audio,
    extract_audio_features,
)


def process_file(file, diagnosis_map):
    parts = file.stem.split("_")
    pid = str(parts[0])

    if pid in DEMO_BLACKLIST:
        print(Fore.BLUE + f"\nSkipped blacklisted patient {pid}" + Style.RESET_ALL)
        return [], []

    df = pd.read_csv(
        file,
        sep="\t",
        names=["start", "end", "crackles", "wheezes"],
    )

    start = df.iloc[0, 0]
    end = df.iloc[-1, 1]
    diagnosis = diagnosis_map.get(pid, "Unknown")

    audio_file = file.with_suffix(".wav")
    audio, sr = lb.load(audio_file, sr=TARGET_SAMPLING_RATE)

    audio_segments = extract_audio_segments(audio, start, end, diagnosis)

    features_list = []
    mel_spec_list = []

    for segment in audio_segments:
        segment_compressed = compress_audio(segment)
        features, mel_spec = extract_audio_features(segment_compressed)

        features = {
            **features,
            "patient_id": pid,
            "diagnosis": diagnosis,
        }

        features_list.append(features)
        mel_spec_list.append(mel_spec)

    return features_list, mel_spec_list


def load_and_preprocess_raw_audio_data():
    raw_audio_path = RAW_AUDIO_PATH

    patient_data = pd.read_csv(DIAGNOSIS_PATH, names=["patient_id", "diagnosis"])
    diagnosis_map = dict(
        zip(patient_data["patient_id"].astype(str), patient_data["diagnosis"])
    )

    files = sorted(raw_audio_path.glob("*.txt"))  # important for order

    results = Parallel(n_jobs=4, backend="loky")(
        delayed(process_file)(file, diagnosis_map) for file in files
    )

    all_features = []
    all_mel_spec = []

    for features_list, mel_spec_list in results:
        all_features.extend(features_list)
        all_mel_spec.extend(mel_spec_list)

    features_df = pd.DataFrame(all_features)
    mel_spec = np.stack(all_mel_spec).astype(np.float32)

    mask = features_df["diagnosis"].isin(CLASS_NAMES).values
    features_filtered = features_df[mask].reset_index(drop=True)
    mel_spec_filtered = mel_spec[mask]

    return features_filtered, mel_spec_filtered
