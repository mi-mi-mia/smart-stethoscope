import os
from pathlib import Path
import pandas as pd
import librosa as lb
import soundfile as sf


def cut_data_into_breathing_cycle(raw_data, start, end, sr=22050):
    """
    Takes a numpy array of raw audio data and spilts it using start and end arguments.

    raw_data=numpy array of audio sample
    start=time
    end=time
    sr=sampling_rate
    """
    max_ind = len(raw_data)
    start_ind = min(int(start * sr), max_ind)
    end_ind = min(int(end * sr), max_ind)
    return raw_data[start_ind:end_ind]


def load_audio_annotations(raw_audio_path: Path) -> pd.DataFrame:
    """
    Takes the path to the raw audio annotation folder. Returns the annotations
    of all annotation files in the folder as a dataframe.

    raw_audio_path=path to the folder where all the raw audio annotation files are

    Returns:
    annotation_data = DataFrame of all audio annotations.
    """
    files_data = []
    for file in raw_audio_path.glob("*.txt"):
        df = pd.read_csv(
            file,
            sep="\t",
            names=["start", "end", "crackles", "wheezes"],
        )
        df["filename"] = file.stem
        files_data.append(df)

    annotation_data = pd.concat(files_data, ignore_index=True)

    # Drop unused columns
    annotation_data = annotation_data.drop(columns=["crackles", "wheezes"])

    return annotation_data


def extract_breathing_cycles():
    """
    Extracts all breathing cycles from all audio files in the raw_data folder according
    to their annotation files and saves the breathing cycles in preprocessed_data folder.
    """

    preproc_audio_path = Path("../preprocessed_data/audio_breathing_cycles/")
    raw_audio_path = Path(
        "../raw_data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"
    )

    Path(preproc_audio_path).mkdir(parents=True, exist_ok=True)

    annotation_data = load_audio_annotations(raw_audio_path)

    # Create cycle index per file
    annotation_data["cycle"] = annotation_data.groupby("filename").cumcount()

    for row in annotation_data.itertuples(index=False):
        audio_file = raw_audio_path / f"{row.filename}.wav"
        save_file = preproc_audio_path / f"{row.filename}_{row.cycle}.wav"

        audio, sr = lb.load(audio_file)
        breathing_cycle = cut_data_into_breathing_cycle(audio, row.start, row.end, sr)

        sf.write(file=save_file, data=breathing_cycle, samplerate=sr)

    print(f"✅ Processed {len(annotation_data)} audio files")
