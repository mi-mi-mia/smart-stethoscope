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
class DataLoader(BaseEstimator, TransformerMixin):
    """
    Loads and merges demographic, patient diagnosis, and audio breath cycle data.

    Parses audio filenames to extract patient ID, chest location.
    Returns one row per breath cycle with all patient metadata merged.
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

        # load and parse audio txt files
        files = [s.split('.')[0] for s in os.listdir(self.audio_path) if '.txt' in s]

        files_data = []
        for file in files:
            data = pd.read_csv(
                self.audio_path + file + '.txt',
                sep='\t',
                names=['start', 'end', 'crackles', 'weezels']
            )
            parts = file.split('_')
            data['pid'] = int(parts[0])
            data['chest_location'] = parts[2]
            data['filename'] = file
            files_data.append(data)

        files_df = pd.concat(files_data).reset_index(drop=True)

        # merge all three
        patient_data['pid'] = patient_data['pid'].astype('int32')
        files_df['pid'] = files_df['pid'].astype('int32')
        demographic_data['pid'] = demographic_data['pid'].astype('int32')

        audio_data = pd.merge(files_df, patient_data, on='pid')
        allfactors_data = pd.merge(audio_data, demographic_data, on='pid')
        return allfactors_data

#stratified group split
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
