import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def preprocess_tabular_data(data: pd.DataFrame):
    # TODO: Add preprocessing pipelines
    # drop underrepresented classes before splitting
    data = data[~data["disease"].isin(["Asthma", "LRTI"])].reset_index(drop=True)
    train_data, test_data = stratified_group_split(data)


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

    # one row per patient with disease label
    patient_diseases = data.groupby("pid")["disease"].first().reset_index()

    # stratify split at patient level
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(
        sss.split(patient_diseases["pid"], patient_diseases["disease"])
    )

    train_pids = patient_diseases.iloc[train_idx]["pid"].values
    test_pids = patient_diseases.iloc[test_idx]["pid"].values

    train_data = data[data["pid"].isin(train_pids)].reset_index(drop=True)
    test_data = data[data["pid"].isin(test_pids)].reset_index(drop=True)

    return train_data, test_data
