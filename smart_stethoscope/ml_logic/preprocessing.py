import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

DISEASE_MAPPING = {
    'Healthy': 0,
    'COPD': 1,
    'URTI': 2,
    'Bronchiectasis': 3,
    'Pneumonia': 4,
    'Bronchiolitis': 5,
}
DISEASES_TO_DROP = {'Asthma', 'LRTI'}


# ─── Transformers ─────────────────────────────────────────────────────────────

class ColumnEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical columns. Run on full data before train/test split.

    - Maps 'disease' to integer using DISEASE_MAPPING
    - Drops rows where disease is in DISEASES_TO_DROP (Asthma, LRTI)
    - Label encodes 'sex' (M=0, F=1)
    - One-hot encodes 'chest_location'

    No statistics are learned from the data, so there is no leakage risk
    running this pre-split.
    """

    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False)
        self.le = LabelEncoder()

    def fit(self, X, y=None):
        X = X[~X['disease'].isin(DISEASES_TO_DROP)]
        self.ohe.fit(X[['chest_location']])
        self.le.fit(X['sex'])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = X[~X['disease'].isin(DISEASES_TO_DROP)].reset_index(drop=True)
        X['disease'] = X['disease'].map(DISEASE_MAPPING)
        X['sex'] = self.le.transform(X['sex'])
        encoded = self.ohe.transform(X[['chest_location']])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.ohe.get_feature_names_out(['chest_location']),
            index=X.index
        )
        X = pd.concat([X, encoded_df], axis=1)
        X = X.drop(columns=['chest_location'])
        return X


class FeatureConstructor(BaseEstimator, TransformerMixin):
    """
    Constructs derived features. Run on full data before train/test split.

    - cycle_length: duration of each breath cycle in seconds (end - start)
    - bmi: normalised BMI from adult or child measurements

    Drops: adult_bmi, child_weight, child_height, start, end.
    Must run after ColumnEncoder (sex must already be integer encoded).
    No leakage risk — all arithmetic, no population statistics.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['cycle_length'] = X['end'] - X['start']
        X['bmi'] = X.apply(calculate_bmi, axis=1)
        X = X.drop(columns=['adult_bmi', 'child_weight', 'child_height', 'start', 'end'])
        return X


class Imputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values. Run post-split, fit on train only.

    - age: mean imputation
    - bmi: mean imputation
    - sex: mode imputation
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


class Scaler(BaseEstimator, TransformerMixin):
    """
    Standard scales continuous features. Run post-split, fit on train only.

    Scales age, bmi, cycle_length to zero mean and unit variance.
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

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


def stratified_group_split(data, test_size=0.2, random_state=42):
    """
    Splits at patient level so no patient appears in both train and test,
    while preserving disease class proportions across the split.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed dataframe containing 'pid' and 'disease' columns.
    test_size : float
        Proportion of patients to include in test set.
    random_state : int

    Returns
    -------
    train_data, test_data : pd.DataFrame
    train_pids, test_pids : np.ndarray
    """
    patient_diseases = data.groupby('pid')['disease'].first().reset_index()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(patient_diseases['pid'], patient_diseases['disease']))
    train_pids = patient_diseases.iloc[train_idx]['pid'].values
    test_pids = patient_diseases.iloc[test_idx]['pid'].values
    train_data = data[data['pid'].isin(train_pids)].reset_index(drop=True)
    test_data = data[data['pid'].isin(test_pids)].reset_index(drop=True)
    return train_data, test_data, train_pids, test_pids

# ─── Main entry point ─────────────────────────────────────────────────────────

def preprocess_tabular_data(data, pipeline_save_path=None):
    """
    Full preprocessing flow for tabular data.

    Pipelines are instantiated fresh each call to avoid state leakage.

    1. Pre-split: encode columns, construct features (no leakage risk)
    2. Stratified group split at patient level
    3. Separate X and y
    4. Post-split: impute and scale (fit on train only)
    5. Optionally save the fitted post-split pipeline

    Parameters
    ----------
    data : pd.DataFrame
        Raw merged dataframe from load_data()
    pipeline_save_path : Path or str, optional
        If provided, saves the fitted post-split pipeline to this path.
        Parent directory will be created if it does not exist.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    train_pids, test_pids : np.ndarray
    """
    pre_split_pipeline = Pipeline([
        ('encode', ColumnEncoder()),
        ('construct', FeatureConstructor()),
    ])

    post_split_pipeline = Pipeline([
        ('impute', Imputer()),
        ('scale', Scaler()),
    ])

    # 1. Pre-split transformations
    data = pre_split_pipeline.fit_transform(data)

    # 2. Train/test split at patient level
    train_data, test_data, train_pids, test_pids = stratified_group_split(data)

    # 3. Separate features and target
    X_train = train_data.drop(columns=['disease', 'pid'])
    y_train = train_data['disease']
    X_test = test_data.drop(columns=['disease', 'pid'])
    y_test = test_data['disease']

    # 4. Post-split: fit on train, transform both
    X_train = post_split_pipeline.fit_transform(X_train)
    X_test = post_split_pipeline.transform(X_test)

    # 5. Optionally save fitted pipeline
    if pipeline_save_path is not None:
        Path(pipeline_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(post_split_pipeline, pipeline_save_path)

    return X_train, X_test, y_train, y_test, train_pids, test_pids #to do change pids to be cycle file name and check how order is preserved


# ==========================
# AUDIO FEATURE EXTRACTION
# ==========================
def extract_mfcc_features(df, audio_folder, n_mfcc=13):
    """
    Extract MFCC summary features for each filename in df.
    Assumes df has 'filename' column and corresponding .wav files exist.
    Calculates mean, std, skew and max.
    """
    mfcc_rows = []

    for filename in df["filename"]:
        file_path = Path(audio_folder) / f"{filename}.wav"

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

        mfcc_rows.append([filename] + list(combined))

    columns = ["filename"]

    for i in range(1, n_mfcc + 1):
        columns.append(f"mfcc_{i}_mean")
    for i in range(1, n_mfcc + 1):
        columns.append(f"mfcc_{i}_std")
    for i in range(1, n_mfcc + 1):
        columns.append(f"mfcc_{i}_skew")
    for i in range(1, n_mfcc + 1):
        columns.append(f"mfcc_{i}_max")

    return pd.DataFrame(mfcc_rows, columns=columns)
