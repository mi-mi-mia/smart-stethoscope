# ================================
# Imports
# ================================
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


CLASS_NAMES = ["Bronchiectasis", "COPD", "Healthy", "Pneumonia", "URTI"]
DEFAULT_XGB_WEIGHT = 0.6


# ================================
# Training models
# ================================
def build_cnn_model(input_shape, num_classes):
    """Build and compile a CNN for mel spectrogram classification."""
    pass

    # return cnn_model


def train_cnn_model():
    pass



# ================================
# Hybrid prediction
# ================================
def predict_xgb_proba(xgb_model, xgb_df):
    """
    Predict class probabilities from the XGBoost model.

    Parameters
    ----------
    xgb_model : fitted XGBoost model
    xgb_df : pd.DataFrame
        Tabular feature dataframe, one row per chunk

    Returns
    -------
    np.ndarray
        Shape: (n_chunks, n_classes)
    """
    return xgb_model.predict_proba(xgb_df)


def predict_cnn_proba(cnn_model, cnn_array):
    """
    Predict class probabilities from the CNN model.

    Parameters
    ----------
    cnn_model : fitted Keras/TensorFlow model
    cnn_array : np.ndarray
        Mel spectrogram array, shape (n_chunks, height, width, channels)

    Returns
    -------
    np.ndarray
        Shape: (n_chunks, n_classes)
    """
    return cnn_model.predict(cnn_array, verbose=0)


def fuse_proba(xgb_proba, cnn_proba, w=DEFAULT_XGB_WEIGHT):
    """
    Fuse XGB and CNN probabilities using weighted average.

    Parameters
    ----------
    xgb_proba : np.ndarray
        Shape: (n_chunks, n_classes)
    cnn_proba : np.ndarray
        Shape: (n_chunks, n_classes)
    w : float
        Weight for XGB. CNN weight becomes (1 - w)

    Returns
    -------
    np.ndarray
        Shape: (n_chunks, n_classes)
    """
    if xgb_proba.shape != cnn_proba.shape:
        raise ValueError(
            f"xgb_proba shape {xgb_proba.shape} does not match "
            f"cnn_proba shape {cnn_proba.shape}"
        )

    return w * xgb_proba + (1 - w) * cnn_proba


def aggregate_chunk_proba(chunk_proba):
    """
    Aggregate chunk-level probabilities into one final probability vector.

    Current approach:
    - mean probability across chunks

    Parameters
    ----------
    chunk_proba : np.ndarray
        Shape: (n_chunks, n_classes)

    Returns
    -------
    np.ndarray
        Shape: (n_classes,)
    """
    if len(chunk_proba.shape) != 2:
        raise ValueError(
            f"Expected 2D array of shape (n_chunks, n_classes), got {chunk_proba.shape}"
        )

    return chunk_proba.mean(axis=0)


def predict_final_class(final_proba, class_names=CLASS_NAMES, threshold=None):
    """
    Convert final probability vector into final class prediction.

    Parameters
    ----------
    final_proba : np.ndarray
        Shape: (n_classes,)
    class_names : list[str] | None
        Optional ordered class names
    threshold : float | None
        If set, predictions below this confidence are labelled "Uncertain"

    Returns
    -------
    dict
        {
            "predicted_index": int,
            "predicted_label": str or None,
            "confidence": float,
            "class_probabilities": dict or list
        }
    """
    predicted_index = int(np.argmax(final_proba))
    confidence = float(final_proba[predicted_index])

    if class_names is not None:
        predicted_label = class_names[predicted_index]
        class_probabilities = {
            class_names[i]: float(final_proba[i])
            for i in range(len(class_names))
        }
    else:
        predicted_label = None
        class_probabilities = final_proba.tolist()

    if threshold is not None and confidence < threshold:
        predicted_label = "Uncertain"

    return {
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "class_probabilities": class_probabilities
    }


def predict_hybrid(xgb_model, cnn_model, xgb_df, cnn_array, w=0.6, class_names=None):
    """
    Full hybrid prediction pipeline:
    - XGB probabilities per chunk
    - CNN probabilities per chunk
    - late fusion
    - aggregate across chunks
    - return final prediction

    Parameters
    ----------
    xgb_model : fitted XGBoost model
    cnn_model : fitted CNN model
    xgb_df : pd.DataFrame
        One row per chunk for XGB
    cnn_array : np.ndarray
        One mel spectrogram per chunk for CNN
    w : float
        Weight for XGB in fusion
    class_names : list[str] | None
        Ordered class names

    Returns
    -------
    dict
        Includes intermediate and final outputs
    """
    xgb_proba = predict_xgb_proba(xgb_model, xgb_df)
    cnn_proba = predict_cnn_proba(cnn_model, cnn_array)

    fused_chunk_proba = fuse_proba(xgb_proba, cnn_proba, w=w)
    final_proba = aggregate_chunk_proba(fused_chunk_proba)
    final_prediction = predict_final_class(final_proba, class_names=class_names)

    return {
        "xgb_chunk_proba": xgb_proba,
        "cnn_chunk_proba": cnn_proba,
        "fused_chunk_proba": fused_chunk_proba,
        "final_proba": final_proba,
        "final_prediction": final_prediction
    }



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
