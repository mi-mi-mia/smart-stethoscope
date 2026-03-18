from pathlib import Path
import pandas as pd
import librosa as lb
import soundfile as sf
from colorama import Fore, Style


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


def extract_breathing_cycles(raw_audio_path: Path, preprocessed_audio_path: Path):
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

    """

    preprocessed_audio_path.mkdir(parents=True, exist_ok=True)

    annotation_data = load_audio_annotations(raw_audio_path)

    print(Fore.BLUE + "\nExtracting respiratory cycles from audio..." + Style.RESET_ALL)
    for row in annotation_data.itertuples(index=False):
        audio_file = raw_audio_path / f"{row.filename}.wav"
        save_file = preprocessed_audio_path / f"{row.cycle_filename}.wav"

        audio, sr = lb.load(audio_file)
        breathing_cycle = cut_audio_data(audio, row.start, row.end, sr)

        sf.write(file=save_file, data=breathing_cycle, samplerate=sr)

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

    cache_path = Path("../preprocessed_data/")
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

    preprocessed_audio_path = Path("preprocessed_data/audio_breathing_cycles/")
    raw_audio_path = Path(
        "raw_data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"
    )
    diagnosis_path = Path(
        "raw_data/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv"
    )
    demographic_data_path = Path("raw_data/demographic_info.txt")

    if not any(preprocessed_audio_path.glob("*.wav")):
        extract_breathing_cycles(
            raw_audio_path=raw_audio_path,
            preprocessed_audio_path=preprocessed_audio_path,
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
