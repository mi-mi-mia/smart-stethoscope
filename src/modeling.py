# ================================
# Imports
# ================================
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ================================
# Baseline training function
# ================================
def run_logistic_baseline(train_df, test_df, target_col="disease", patient_col="patient_id"):
    """
    Train and evaluate a multiclass logistic regression model
    using pre-split train and test data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
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
    cols_to_drop = [target_col]

    # Drop patient_id if present
    if patient_col in train_df.columns:
        cols_to_drop.append(patient_col)

    # Drop filename too if present
    if "file_name" in train_df.columns:
        cols_to_drop.append("file_name")

    if "filename" in train_df.columns:
        cols_to_drop.append("filename")

   # -----------------------------
    # Define features and target
    # -----------------------------
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df[target_col]

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
