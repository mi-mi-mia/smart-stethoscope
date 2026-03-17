# ================================
# Imports
# ================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ================================
# Baseline training function
# ================================
def run_logistic_baseline(df, target_col="disease", patient_col="patient_id"):
    """
    Train and evaluate a multiclass logistic regression model
    using a patient-level train/test split to avoid leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Clean modelling dataframe containing:
        - target column
        - patient_id column
        - file_name column (optional, dropped if present)
        - all feature columns already numeric
    target_col : str
        Name of target column
    patient_col : str
        Name of patient identifier column

    Returns
    -------
    model : trained LogisticRegression model
    scaler : fitted StandardScaler
    X_train, X_test, y_train, y_test, y_pred : useful outputs for later inspection
    """

    # -----------------------------
    # Define target
    # -----------------------------
    y = df[target_col]

    # -----------------------------
    # Define features
    # Drop target and identifier columns
    # -----------------------------
    cols_to_drop = [target_col, patient_col]

    # Drop file_name too if it exists, because it is an identifier not a feature
    if "file_name" in df.columns:
        cols_to_drop.append("file_name")

    X = df.drop(columns=cols_to_drop)

    # -----------------------------
    # Patient-level train/test split
    # -----------------------------
    unique_patients = df[patient_col].unique()

    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=0.2,
        random_state=42
    )

    # Boolean masks used to keep all rows for a patient together
    train_mask = df[patient_col].isin(train_patients)
    test_mask = df[patient_col].isin(test_patients)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    # -----------------------------
    # Scale numeric features
    # Fit on train only to avoid leakage
    # -----------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # Train model
    # -----------------------------
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    model.fit(X_train_scaled, y_train)

    # -----------------------------
    # Predict
    # -----------------------------
    y_pred = model.predict(X_test_scaled)

    # -----------------------------
    # Evaluate
    # -----------------------------
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    # -----------------------------
    # Return useful objects
    # -----------------------------
    return model, scaler, X_train, X_test, y_train, y_test, y_pred
