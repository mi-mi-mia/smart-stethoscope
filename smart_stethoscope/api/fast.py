import os
import io
import json
import numpy as np
import pandas as pd
import librosa
import joblib
from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
from google.cloud import storage
from smart_stethoscope.ml_logic.preprocessing import audio_preprocessing
from smart_stethoscope.ml_logic.model import predict_hybrid

# ================================
# GCS loading helpers
# ================================
def load_pickle_from_gcs(gcs_path: str):
    """Load a pickle file from GCS into memory."""
    client = storage.Client()
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    return joblib.load(buffer)

def load_json_from_gcs(gcs_path: str):
    """Load a JSON file from GCS into memory."""
    client = storage.Client()
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return json.loads(blob.download_as_text())

# ================================
# Model loading
# ================================
CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH", "gs://smart-stethoscope/cnn_model.keras")
XGB_MODEL_PATH = os.getenv("XGB_MODEL_PATH", "gs://smart-stethoscope/xgb_model.pkl")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "gs://smart-stethoscope/feature_columns.pkl")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "gs://smart-stethoscope/class_names.json")

cnn_model = keras.models.load_model(CNN_MODEL_PATH)
xgb_model = load_pickle_from_gcs(XGB_MODEL_PATH)
feature_columns = load_pickle_from_gcs(FEATURE_COLUMNS_PATH)
CLASS_NAMES = load_json_from_gcs(CLASS_NAMES_PATH)

# ================================
# App
# ================================
app = FastAPI()

@app.get("/")
def index():
    return {"status": "API is online"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cnn_model_loaded": cnn_model is not None,
        "xgb_model_loaded": xgb_model is not None,
        "feature_columns_loaded": feature_columns is not None,
        "class_names": CLASS_NAMES
    }

@app.post("/predict")
async def predict_audio(
    audio_file: UploadFile = File(...),
    annotation_file: UploadFile = File(...)
):
    # 1. Read both files into memory as bytes
    audio_bytes = await audio_file.read()
    annotation_bytes = await annotation_file.read()

    # 2. Load audio into numpy array — no disk write needed
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # 3. Parse annotation .txt into a DataFrame
    annotations = pd.read_csv(
        io.StringIO(annotation_bytes.decode("utf-8")),
        sep="\t",
        names=["start", "end", "crackles", "wheezes"]
    )

    # 4. Preprocess each cycle — returns tabular features and mel spectrograms
    features_list = []
    mel_spec_list = []

    for _, row in annotations.iterrows():
        features_df, mel_spec = audio_preprocessing(audio, sr, row["start"], row["end"])
        if mel_spec.shape[0] > 0:
            features_list.append(features_df)
            mel_spec_list.append(mel_spec)

    # 5. Stack all cycles into arrays for hybrid model
    xgb_features = pd.concat(features_list, ignore_index=True)
    cnn_features = np.concatenate(mel_spec_list, axis=0)

    # 6. Align XGB feature columns to match training order
    xgb_features = xgb_features[feature_columns]

    # 7. Run hybrid prediction
    result = predict_hybrid(
        xgb_model=xgb_model,
        cnn_model=cnn_model,
        xgb_df=xgb_features,
        cnn_array=cnn_features,
        class_names=CLASS_NAMES
    )

    final = result["final_prediction"]

    return {
        "prediction": final["predicted_label"],
        "confidence": final["confidence"],
        "class_probabilities": final["class_probabilities"],
        "cycles_analysed": len(mel_spec_list)
    }
