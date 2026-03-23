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


# ================================
# Simple CNN
# ================================
def build_cnn_model(input_shape, num_classes):
    """Build and compile a small CNN for mel spectrogram classification."""

    cnn_model = models.Sequential()

    cnn_model.add(layers.Input(shape=input_shape))
    cnn_model.add(
        layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )
    )
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )
    )
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(layers.GlobalAveragePooling2D())
    cnn_model.add(layers.Dense(32, activation="relu"))
    cnn_model.add(layers.Dropout(0.3))
    cnn_model.add(layers.Dense(num_classes, activation="softmax"))

    cnn_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return cnn_model


def train_cnn_model(
    X_train_img,
    X_test_img,
    y_train,
    y_test,
    epochs=15,
    batch_size=32,
    patience=5
):
    """
    Train and evaluate a CNN on mel spectrogram images.

    Parameters
    ----------
    X_train_img : np.ndarray
        Training spectrogram images
    X_test_img : np.ndarray
        Test spectrogram images
    y_train : pd.Series or np.ndarray
        Integer class labels for training
    y_test : pd.Series or np.ndarray
        Integer class labels for test

    Returns
    -------
    cnn_model : keras.Model
        Trained CNN model
    history : keras.callbacks.History
        Training history
    metrics : dict
        Summary evaluation metrics
    y_pred_cnn : np.ndarray
        Predicted class labels for test set
    """

    # CNN-specific label prep
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    input_shape = X_train_img.shape[1:]

    # Build model
    cnn_model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)

    # Class weights
    y_train_int = np.argmax(y_train_cat, axis=1)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    # Train
    history = cnn_model.fit(
        X_train_img,
        y_train_cat,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate
    test_loss, test_accuracy = cnn_model.evaluate(X_test_img, y_test_cat, verbose=1)

    y_pred_probs = cnn_model.predict(X_test_img)
    y_pred_cnn = np.argmax(y_pred_probs, axis=1)
    y_test_int = np.argmax(y_test_cat, axis=1)

    macro_f1 = f1_score(y_test_int, y_pred_cnn, average="macro")

    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)
    print("Macro F1:", macro_f1)
    print(classification_report(y_test_int, y_pred_cnn, zero_division=0))

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(macro_f1),
        "num_classes": int(num_classes)
    }

    return cnn_model, history, metrics, y_pred_cnn



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
