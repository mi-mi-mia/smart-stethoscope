from pathlib import Path
from colorama import Fore, Style
import pandas as pd
import numpy as np
import librosa as lb
import soundfile as sf
from smart_stethoscope.params import *
from smart_stethoscope.ml_logic.audio_preprocessing import cut_audio_data, pad_audio


def load_audio_annotations(raw_audio_path: Path) -> pd.DataFrame:
    """
    Takes the path to the raw audio annotation folder. Returns the annotations
    of all annotation files in the folder as a dataframe including the information
    from the file name.

    Parameters
    ----------
    raw_audio_path : Path
        Path to the folder where all the raw audio annotation files are

    Returns
    -------
    annotation_data : pd.dataFrame
        DataFrame of all audio annotations.
    """
    files_data = []
    for file in raw_audio_path.glob("*.txt"):
        df = pd.read_csv(
            file,
            sep="\t",
            names=["start", "end", "crackles", "wheezes"],
        )
        df["filename"] = file.stem
        parts = file.stem.split("_")
        df["pid"] = int(parts[0])
        df["chest_location"] = parts[2]
        files_data.append(df)

    annotation_data = pd.concat(files_data, ignore_index=True)
    # Create cycle filename
    annotation_data["cycle_filename"] = (
        annotation_data["filename"].astype(str)
        + "_"
        + annotation_data.groupby("filename").cumcount().astype(str)
    )

    return annotation_data


def extract_breathing_cycles(
    raw_audio_path: Path, preprocessed_audio_path: Path, preprocessed_padded_audio_path
):
    """
    Extracts and saves individual breathing cycles from raw audio files.

    Slices each breathing cycle from the raw .wav file according to its
    annotation and saves as a new .wav file.

    Parameters
    ----------
    raw_audio_path : Path
        Path to the folder where all the raw audio files are
    preprocessed_audio_path : Path
        Path to the folder where the extracted breathing cycles should be saved
    preprocessed_padded_audio_path : Path
        Path to the folder where the padded breathing cycles should be saved

    """

    preprocessed_audio_path.mkdir(parents=True, exist_ok=True)
    preprocessed_padded_audio_path.mkdir(parents=True, exist_ok=True)

    annotation_data = load_audio_annotations(raw_audio_path)

    print(Fore.BLUE + "\nExtracting respiratory cycles from audio..." + Style.RESET_ALL)
    for row in annotation_data.itertuples(index=False):
        audio_file = raw_audio_path / f"{row.filename}.wav"
        save_file = preprocessed_audio_path / f"{row.cycle_filename}.wav"
        save_padding_audio = (
            preprocessed_padded_audio_path / f"{row.cycle_filename}.wav"
        )

        # TODO: Add once we decide for compression and remove lb loading.
        # audio, sr = sf.read(audio_file, dtype="float32")
        # mu = 255
        # audio_compressed = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)

        audio, sr = lb.load(audio_file, sr=None)
        audio = lb.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLING_RATE)

        breathing_cycle = cut_audio_data(
            audio, row.start, row.end, TARGET_SAMPLING_RATE
        )

        sf.write(file=save_file, data=breathing_cycle, samplerate=sr)

        padded_data = pad_audio(breathing_cycle)
        sf.write(
            file=save_padding_audio, data=padded_data, samplerate=TARGET_SAMPLING_RATE
        )

    print(
        Fore.BLUE
        + f"\n✅ Processed {len(annotation_data)} audio files"
        + Style.RESET_ALL
    )


def load_tabular_data(
    demographic_data_path: Path, diagnosis_path: Path, raw_audio_path: Path
) -> pd.DataFrame:
    """
    Loads patient demographics, patient diagnosis, and audio annotations.
    If it already exists, loads from cach, else from raw data.

    Parameters
    ----------
    demographic_data_path : Path
        Path to the to the demographic data file
    diagnosis_path : Path
        Path to the to the diagnosis data file
    raw_audio_path : Path
        Path to the folder where all the raw audio files are

    Returns
    -------
    allfactors_data : DataFrame
        Data frame of all raw tabular data.
    """

    cache_path = CACHE_PATH
    cache_file = cache_path / "raw_tabular_data.csv"
    if cache_file.is_file():
        print(Fore.BLUE + "\nLoad data from cached CSV..." + Style.RESET_ALL)
        allfactors_data = pd.read_csv(cache_file, header="infer")

    else:
        print(Fore.BLUE + "\nLoad data from raw data folder..." + Style.RESET_ALL)
        # load demographic data
        demographic_data = pd.read_csv(
            demographic_data_path,
            sep=" ",
            header=None,
            names=["pid", "age", "sex", "adult_bmi", "child_weight", "child_height"],
        )

        # load patient diagnosis

        patient_data = pd.read_csv(diagnosis_path, names=["pid", "disease"])

        # load file annotations
        audio_annotations = load_audio_annotations(raw_audio_path)

        patient_data["pid"] = patient_data["pid"].astype("int32")
        audio_annotations["pid"] = audio_annotations["pid"].astype("int32")
        demographic_data["pid"] = demographic_data["pid"].astype("int32")

        audio_data = pd.merge(audio_annotations, patient_data, on="pid")
        allfactors_data = pd.merge(audio_data, demographic_data, on="pid")
        allfactors_data = allfactors_data.drop(columns=["pid"])  # add here

        # Save tabular data in cache
        cache_path.mkdir(parents=True, exist_ok=True)
        allfactors_data.to_csv(cache_file, index=False)

    return allfactors_data


def load_data() -> pd.DataFrame:
    """
    Extracts breathing cycles from raw audio data and loads patient demographics,
    diagnosis, and audio file annotations.

    Returns
    -------
    raw_tabular_data : DataFrame
        Data frame of all raw tabular data.
    """

    preprocessed_audio_path = PREPROCESSED_AUDIO_PATH
    preprocessed_padded_audio_path = PREPROCESSED_PADDED_AUTIO_PATH
    raw_audio_path = RAW_AUDIO_PATH
    diagnosis_path = DIAGNOSIS_PATH
    demographic_data_path = DEMOGRAPHIC_DATA_PATH

    if not any(preprocessed_audio_path.glob("*.wav")):
        extract_breathing_cycles(
            raw_audio_path=raw_audio_path,
            preprocessed_audio_path=preprocessed_audio_path,
            preprocessed_padded_audio_path=preprocessed_padded_audio_path,
        )
    else:
        print(
            Fore.BLUE
            + "\n✅ Processed audio files already exist, skipping extraction"
            + Style.RESET_ALL
        )

    raw_tabular_data = load_tabular_data(
        demographic_data_path, diagnosis_path, raw_audio_path
    )
    return raw_tabular_data
