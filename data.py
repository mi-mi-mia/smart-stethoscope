#data loading definitions

import numpy as pd
import pandas as pd
from pathlib import Path
import librosa as lb
import soundfile as sf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit

#step 1 - run once to produce audio files
def get_breathing_cycle(raw_data, start, end, sr=22050):
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
def extract_breathing_cycles(raw_audio_path, preproc_audio_path, max_len=6):
    """
    Extracts and saves individual breathing cycles from raw audio files.

    Slices each breathing cycle from the raw .wav file according to its
    annotation, pads to max_len seconds, and saves as a new .wav file.
    Must be run before DataLoader as it produces the processed audio files
    that audio feature extraction will consume.

    Parameters
    ----------
    raw_audio_path : str or Path
        Path to folder containing raw .wav and .txt annotation files
    preproc_audio_path : str or Path
        Path to folder where processed breathing cycle .wav files will be saved
    max_len : int
        Maximum length of each breathing cycle in seconds (default 6)
    """
    raw_audio_path = Path(raw_audio_path)
    preproc_audio_path = Path(preproc_audio_path)
    preproc_audio_path.mkdir(parents=True, exist_ok=True)

    # load annotations first so we know what to extract
    files_data = []
    for file in raw_audio_path.glob("*.txt"):
        df = pd.read_csv(file, sep="\t", names=["start", "end", "crackles", "weezels"])
        df["filename"] = file.stem
        files_data.append(df)

    annotation_data = pd.concat(files_data, ignore_index=True)
    annotation_data["cycle"] = annotation_data.groupby("filename").cumcount()

    for row in annotation_data.itertuples(index=False):
        audio_file = raw_audio_path / f"{row.filename}.wav"
        save_file = preproc_audio_path / f"{row.filename}_{row.cycle}.wav"

        audio, sr = lb.load(audio_file)

        # cap cycle length at max_len
        end = min(row.end, row.start + max_len)
        breathing_cycle = get_breathing_cycle(audio, row.start, end, sr)

        # pad to max_len so all cycles are same length
        req_len = max_len * sr
        padded_cycle = lb.util.pad_center(breathing_cycle, size=req_len)

        sf.write(file=save_file, data=padded_cycle, samplerate=sr)

    print(f"✅ Extracted {len(annotation_data)} breathing cycles")
    return annotation_data

#step 2 build tabular feature dataframe
class DataLoader(BaseEstimator, TransformerMixin):
    """
    Loads and merges annotation, demographic, and patient diagnosis data.

    Uses pathlib.glob for cleaner file handling (from partner's implementation).
    Keeps crackles and weezels as features.
    Extracts pid and chest_location from filename metadata.
    Returns one row per breath cycle with all patient metadata merged.

    Must be run after extract_breathing_cycles() so processed audio files exist
    for downstream audio feature extraction.
    """

    def __init__(self, audio_path, demographic_path, diagnosis_path):
        self.audio_path = audio_path
        self.demographic_path = demographic_path
        self.diagnosis_path = diagnosis_path

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        # load demographic data
        demographic_data = pd.read_csv(
            self.demographic_path,
            sep=' ',
            header=None,
            names=["pid", "age", "sex", "adult_bmi", "child_weight", "child_height"]
        )

        # load patient diagnosis
        patient_data = pd.read_csv(
            self.diagnosis_path,
            names=['pid', 'disease']
        )

        # load annotations using pathlib - cleaner than os.listdir
        files_data = []
        for file in Path(self.audio_path).glob("*.txt"):
            df = pd.read_csv(
                file,
                sep="\t",
                names=['start', 'end', 'crackles', 'weezels']
            )
            parts = file.stem.split('_')
            df['pid'] = int(parts[0])
            df['chest_location'] = parts[2]
            df['filename'] = file.stem
            files_data.append(df)

        files_df = pd.concat(files_data, ignore_index=True)

        # merge all three
        patient_data['pid'] = patient_data['pid'].astype('int32')
        files_df['pid'] = files_df['pid'].astype('int32')
        demographic_data['pid'] = demographic_data['pid'].astype('int32')

        audio_data = pd.merge(files_df, patient_data, on='pid')
        allfactors_data = pd.merge(audio_data, demographic_data, on='pid')

        return allfactors_data

#step 3 stratified group split
def stratified_group_split(data, test_size=0.2, random_state=42):
    """
    Splits data into train and test sets ensuring:
    - No patient appears in both train and test
    - Disease class proportions are preserved across the split

    Parameters
    ----------
    data : pd.DataFrame
        Raw merged dataframe containing 'pid' and 'disease' columns
    test_size : float
        Proportion of patients to include in test set
    random_state : int

    Returns
    -------
    train_data, test_data : pd.DataFrame
    """
    # drop underrepresented classes before splitting
    data = data[~data['disease'].isin(['Asthma', 'LRTI'])].reset_index(drop=True)

    # one row per patient with disease label
    patient_diseases = data.groupby('pid')['disease'].first().reset_index()

    # stratify split at patient level
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(patient_diseases['pid'], patient_diseases['disease']))

    train_pids = patient_diseases.iloc[train_idx]['pid'].values
    test_pids = patient_diseases.iloc[test_idx]['pid'].values

    train_data = data[data['pid'].isin(train_pids)].reset_index(drop=True)
    test_data = data[data['pid'].isin(test_pids)].reset_index(drop=True)

    return train_data, test_data
