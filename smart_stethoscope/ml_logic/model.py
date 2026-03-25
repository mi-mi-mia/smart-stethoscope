# ================================
# Imports
# ================================
import numpy as np
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from smart_stethoscope.params import CLASS_NAMES


DEFAULT_XGB_WEIGHT = 0.8


# ================================
# GROUP SPLITS
# ================================
def make_group_train_val_split(X, y, groups, n_splits=3, random_state=42):
    """
    Create one stratified grouped train/validation split.
    - grouped by patient
    - stratified by class
    - one train/val split only
    - no test split here

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Integer labels
    groups : pd.Series or np.ndarray
        Group labels (e.g. patient IDs)
    n_splits : int
        Number of SGKF splits
    random_state : int
        Random seed

    Returns
    -------
    train_idx : np.ndarray
        Training indices
    val_idx : np.ndarray
        Validation indices
    """
    y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    groups_array = (
        groups.to_numpy() if hasattr(groups, "to_numpy") else np.asarray(groups)
    )

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    train_idx, val_idx = next(sgkf.split(X, y_array, groups_array))
    return train_idx, val_idx


# ================================
# Build and train individual models
# ================================
def build_xgb_model(num_classes: int) -> xgb.XGBClassifier:
    """
    Build XGBoost model.

    Parameters
    ----------
    num_classes : int
        Number of target classes

    Returns
    -------
    xgb.XGBClassifier
        Untrained XGBoost classifier
    """
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.5,
        objective="multi:softprob",
        num_class=num_classes,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=15,
    )
    return model


def train_xgb_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train XGBoost model using provided train/validation split.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training labels
    X_val : pd.DataFrame or np.ndarray, optional
        Validation features for early stopping
    y_val : pd.Series or np.ndarray, optional
        Validation labels for early stopping

    Returns
    -------
    xgb.XGBClassifier
        Trained XGBoost model
    """
    num_classes = len(np.unique(y_train))
    xgb_model = build_xgb_model(num_classes=num_classes)

    # Balance classes on training set only
    w_train = compute_sample_weight(class_weight="balanced", y=y_train)

    # Use validation set if provided
    if X_val is not None and y_val is not None:
        xgb_model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        # Full-data training path (e.g. final cloud training)
        # Fallback path if no validation set is provided
        xgb_model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

    return xgb_model


def build_cnn_model(input_shape, num_classes):
    """
    Build and compile CNN.

    Parameters
    ----------
    input_shape : tuple
        Shape of one mel spectrogram sample, e.g. (128, 141, 1)
    num_classes : int
        Number of target classes

    Returns
    -------
    keras.Model
        Compiled CNN model
    """
    cnn_model = models.Sequential()

    # Input
    cnn_model.add(layers.Input(shape=input_shape))

    # Conv2D Block 1
    cnn_model.add(layers.Conv2D(32, (3, 3), padding="same"))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.Activation("relu"))
    cnn_model.add(layers.MaxPooling2D((2, 2)))

    # Conv2D Block 2
    cnn_model.add(layers.Conv2D(64, (3, 3), padding="same"))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.Activation("relu"))
    cnn_model.add(layers.MaxPooling2D((2, 2)))

    # Conv2D Block 3
    cnn_model.add(layers.Conv2D(128, (3, 3), padding="same"))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.Activation("relu"))
    cnn_model.add(layers.MaxPooling2D((2, 2)))

    # Conv2D Block 4
    cnn_model.add(layers.Conv2D(256, (3, 3), padding="same"))
    cnn_model.add(layers.BatchNormalization())
    cnn_model.add(layers.Activation("relu"))
    cnn_model.add(layers.MaxPooling2D((2, 2)))

    # Turn feature maps into one vector
    cnn_model.add(layers.GlobalMaxPooling2D())

    # Dense layer before classification
    cnn_model.add(layers.Dense(32, activation="relu"))
    cnn_model.add(layers.Dropout(0.3))

    # Final prediction layer
    cnn_model.add(layers.Dense(num_classes, activation="softmax"))

    cnn_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return cnn_model


def train_cnn_model(
    X_train_img,
    y_train,
    X_val_img=None,
    y_val=None,
    epochs=30,
    batch_size=32,
    patience=5,
):
    """
    Train CNN model.

    Parameters
    ----------
    X_train_img : np.ndarray
        Training mel spectrogram array
    y_train : np.ndarray or pd.Series
        Integer labels for training
    X_val_img : np.ndarray, optional
        Validation mel spectrogram array
    y_val : np.ndarray or pd.Series, optional
        Integer labels for validation
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    patience : int
        Early stopping patience

    Returns
    -------
    keras.Model
        Trained CNN model
    """
    num_classes = len(np.unique(y_train))

    # One-hot encode
    y_train_cat = to_categorical(y_train, num_classes)
    input_shape = X_train_img.shape[1:]

    cnn_model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)

    callbacks = [EarlyStopping(patience=patience, restore_best_weights=True)]

    # Validation-data training path
    if X_val_img is not None and y_val is not None:
        y_val_cat = to_categorical(y_val, num_classes)

        cnn_model.fit(
            X_train_img,
            y_train_cat,
            validation_data=(X_val_img, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
    else:
        # Full-data training path (e.g. final cloud training)
        # Fallback path if no validation set is provided
        cnn_model.fit(
            X_train_img, y_train_cat, epochs=epochs, batch_size=batch_size, verbose=0
        )

    return cnn_model


# ================================
# Train the hybrid model
# ================================
def train_final_hybrid_models(
    X, y, mel_spectrograms, groups, n_splits=3, random_state=42
):
    """
    Train final XGB and CNN models using one stratified grouped train/val split.
    No test split here - this is for FINAL MODEL training only.
    """

    # -----------------------------
    # Shared grouped train/val split
    # -----------------------------
    train_idx, val_idx = make_group_train_val_split(
        X=X, y=y, groups=groups, n_splits=n_splits, random_state=random_state
    )

    # -----------------------------
    # XGB inputs
    # -----------------------------
    if hasattr(X, "iloc"):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
    else:
        X_train = X[train_idx]
        X_val = X[val_idx]

    y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    if hasattr(y, "iloc"):
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
    else:
        y_train = y_array[train_idx]
        y_val = y_array[val_idx]

    # -----------------------------
    # CNN inputs (same indices)
    # -----------------------------
    X_train_img = mel_spectrograms[train_idx]
    X_val_img = mel_spectrograms[val_idx]

    print("Train classes:", np.sort(np.unique(y_train)))
    print("Val classes:  ", np.sort(np.unique(y_val)))
    print("Train X shape:", X_train.shape)
    print("Val X shape:  ", X_val.shape)
    print("Train mel shape:", X_train_img.shape)
    print("Val mel shape:  ", X_val_img.shape)

    # -----------------------------
    # Train XGB
    # -----------------------------
    xgb_model = train_xgb_model(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
    )

    # -----------------------------
    # Train CNN
    # -----------------------------
    cnn_model = train_cnn_model(
        X_train_img=X_train_img, y_train=y_train, X_val_img=X_val_img, y_val=y_val
    )

    return {
        "xgb_model": xgb_model,
        "cnn_model": cnn_model,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


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
            class_names[i]: float(final_proba[i]) for i in range(len(class_names))
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
        "class_probabilities": class_probabilities,
    }


def predict_hybrid(
    xgb_model,
    cnn_model,
    xgb_df,
    cnn_array,
    w=DEFAULT_XGB_WEIGHT,
    class_names=CLASS_NAMES,
):
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
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(predict_xgb_proba, xgb_model, xgb_df),
            executor.submit(predict_cnn_proba, cnn_model, cnn_array),
        ]
        results = [f.result() for f in futures]
    xgb_proba, cnn_proba = results

    fused_chunk_proba = fuse_proba(xgb_proba, cnn_proba, w=w)
    final_proba = aggregate_chunk_proba(fused_chunk_proba)
    final_prediction = predict_final_class(final_proba, class_names=class_names)

    return {
        "xgb_chunk_proba": xgb_proba,
        "cnn_chunk_proba": cnn_proba,
        "fused_chunk_proba": fused_chunk_proba,
        "final_proba": final_proba,
        "final_prediction": final_prediction,
    }
