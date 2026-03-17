#imports
import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#load data
import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa as lb
import soundfile as sf
from sklearn.base import BaseEstimator, TransformerMixin

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


#step 4 actual preprocessing (encode, scale etc)
#encode features - sex, disease, chest location
class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features and drops underrepresented disease classes.

    Steps:
    - Label encodes sex (M=0, F=1)
    - One hot encodes disease and chest_location

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame with encoded features and original categorical columns dropped.
    """

    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False)
        self.le = LabelEncoder()

    def fit(self, X, y=None):
        self.ohe.fit(X[['disease', 'chest_location']])
        self.le.fit(X['sex'])
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # label encode sex
        X['sex'] = self.le.transform(X['sex'])

        # one hot encode disease and chest_location
        encoded = self.ohe.transform(X[['disease', 'chest_location']])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.ohe.get_feature_names_out(['disease', 'chest_location']),
            index=X.index
        )

        X = pd.concat([X, encoded_df], axis=1)
        X = X.drop(columns=['disease', 'chest_location'])

        return X


class FeatureConstructor(BaseEstimator, TransformerMixin):
    """
    Constructs derived features:
    - cycle_length: duration of each breath cycle in seconds
    - bmi: normalised BMI from adult or child measurements

    Drops original columns: adult_bmi, child_weight, child_height, start, end.
    Must run after FeatureEncoder so sex is already label encoded (M=0, F=1).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['cycle_length'] = X['end'] - X['start']
        X['bmi'] = X.apply(calculate_bmi, axis=1)
        X = X.drop(columns=['adult_bmi', 'child_weight', 'child_height', 'start', 'end'])
        return X
def calculate_bmi(row):
    """
    Calculate a normalised BMI percentage for a patient row.

    For adults (age >= 19), returns adult BMI normalised against a reference
    value of 25 (WHO healthy BMI upper threshold).

    For children, calculates raw BMI from weight and height, then normalises
    against age- and sex-specific median BMI reference values. Reference values
    are approximated from CDC growth chart data.

    Parameters
    ----------
    row : pd.Series
        A single row from a DataFrame containing:
        - age (float): patient age in years
        - sex (int or str): 0/'M' = Male, 1/'F' = Female
        - adult_bmi (float): BMI for adults, NaN for children
        - child_weight (float): weight in kg for children, NaN for adults
        - child_height (float): height in cm for children, NaN for adults

    Returns
    -------
    float
        Normalised BMI percentage, or NaN if required fields are missing.

    Notes
    -----
    Age thresholds are coarse approximations. Finer age resolution would
    improve accuracy for children, particularly around puberty.
    """
    if row['age'] >= 19:
        return row['adult_bmi'] / 25 if pd.notna(row['adult_bmi']) else np.nan

    if pd.isna(row['child_weight']) or pd.isna(row['child_height']):
        return np.nan

    raw_bmi = row['child_weight'] / (row['child_height'] / 100) ** 2
    age = row['age']
    sex = row['sex']

    # handle both encoded (0/1) and raw ('M'/'F') sex values
    is_male = sex in (0, 'M')
    is_female = sex in (1, 'F')

    if age < 2:
        return raw_bmi / 16.5
    elif is_male:
        thresholds = [(9, 16), (11, 17), (12, 18), (13, 18.6), (14, 19.3),
                      (15, 20), (16, 20.6), (17, 21.3), (18, 22), (19, 22.6), (20, 23)]
        for max_age, divisor in thresholds:
            if age <= max_age:
                return raw_bmi / divisor
    elif is_female:
        thresholds = [(9, 16), (11, 17.5), (12, 18), (13, 18.6), (14, 19.3),
                      (15, 20), (16, 20.5), (17, 20.8), (18, 21.2), (19, 21.5), (20, 21.8)]
        for max_age, divisor in thresholds:
            if age <= max_age:
                return raw_bmi / divisor

    return np.nan

#impute age, bmi, sex
class Imputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values for age, bmi and sex.

    - age: mean imputation
    - bmi: mean imputation
    - sex: mode imputation

    Must run after train/test split to avoid data leakage.
    Fit on training data only, then transform both train and test.

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame with missing values imputed.
    """

    def __init__(self):
        self.mean_imputer = SimpleImputer(strategy='mean')
        self.mode_imputer = SimpleImputer(strategy='most_frequent')

    def fit(self, X, y=None):
        self.mean_imputer.fit(X[['age', 'bmi']])
        self.mode_imputer.fit(X[['sex']])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[['age', 'bmi']] = self.mean_imputer.transform(X[['age', 'bmi']])
        X[['sex']] = self.mode_imputer.transform(X[['sex']])
        return X


#scale age, bmi, cycle length
class Scaler(BaseEstimator, TransformerMixin):
    """
    Applies standard scaling to continuous features: age, bmi, cycle_length.

    Scales to zero mean and unit variance.
    Must run after train/test split to avoid data leakage.
    Fit on training data only, then transform both train and test.

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame with scaled continuous features.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[['age', 'bmi', 'cycle_length']])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[['age', 'bmi', 'cycle_length']] = self.scaler.transform(X[['age', 'bmi', 'cycle_length']])
        return X
