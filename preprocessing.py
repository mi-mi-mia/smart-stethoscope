#imports
import numpy as np
import pandas as pd

import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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

#encode features - sex, disease, chest lo
class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features and drops underrepresented disease classes.

    Steps:
    - Drops rows where disease is Asthma or LRTI (insufficient samples)
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
        X_filtered = X[~X['disease'].isin(['Asthma', 'LRTI'])]
        self.ohe.fit(X_filtered[['disease', 'chest_location']])
        self.le.fit(X_filtered['sex'])
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # drop underrepresented classes
        X = X[~X['disease'].isin(['Asthma', 'LRTI'])].reset_index(drop=True)

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

#feature construction - BMI
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
        - sex (int): 0 = Male, 1 = Female
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
    if row['age'] >= 19:  # adult
        return row['adult_bmi'] / 25 if pd.notna(row['adult_bmi']) else np.nan

    if pd.isna(row['child_weight']) or pd.isna(row['child_height']):
        return np.nan

    raw_bmi = row['child_weight'] / (row['child_height'] / 100) ** 2
    age = row['age']
    sex = row['sex']  # M=0, F=1

    if age < 2:
        return raw_bmi / 16.5
    elif sex == 0:  # Male
        thresholds = [(9, 16), (11, 17), (12, 18), (13, 18.6), (14, 19.3),
                      (15, 20), (16, 20.6), (17, 21.3), (18, 22), (19, 22.6), (20, 23)]
        for max_age, divisor in thresholds:
            if age <= max_age:
                return raw_bmi / divisor
    elif sex == 1:  # Female
        thresholds = [(9, 16), (11, 17.5), (12, 18), (13, 18.6), (14, 19.3),
                        (15, 20), (16, 20.5), (17, 20.8), (18, 21.2), (19, 21.5), (20, 21.8)]
        for max_age, divisor in thresholds:
            if age <= max_age:
                return raw_bmi / divisor
    return np.nan

class BMICalculator(BaseEstimator, TransformerMixin):
    """
    Combines adult BMI and child height/weight into a single normalised BMI column.

    Applies calculate_bmi row-wise, then drops the original adult_bmi,
    child_weight and child_height columns.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['bmi'] = X.apply(calculate_bmi, axis=1)
        X = X.drop(columns=['adult_bmi', 'child_weight', 'child_height'])
        return X
