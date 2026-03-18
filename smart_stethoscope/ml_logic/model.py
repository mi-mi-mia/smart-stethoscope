# ================================
# Imports
# ================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ================================
# Baseline training function
# ================================
def run_logistic_baseline(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a multiclass logistic regression model
    using already-preprocessed train and test data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Preprocessed training features
    X_test : pd.DataFrame
        Preprocessed test features
    y_train : pd.Series
        Training target
    y_test : pd.Series
        Test target

    Returns
    -------
    model : trained LogisticRegression model
    y_pred : np.ndarray
        Predicted labels for X_test
    """

    # -----------------------------
    # Train model
    # -----------------------------
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Predict
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # Evaluate
    # -----------------------------
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    # -----------------------------
    # Return useful objects
    # -----------------------------
    return model, y_pred
