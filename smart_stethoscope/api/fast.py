import os
import io
import json
import numpy as np
import pandas as pd
import librosa
import pickle
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from smart_stethoscope.interface.main import preprocess_for_prediction, predict

# Load once at startup
CNN_MODEL_PATH = os.getenv("MODEL_PATH", "gs://smart-stethoscope/cnn_model.keras")
XGB_MODEL_PATH = os.getenv("MODEL_PATH", "gs://smart-stethoscope/xgb_model.pkl")
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
with open(XGB_MODEL_PATH, "rb") as f:
    xgb_model = pickle.load(f)

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

    # 3. Parse annotation .txt into DataFrame
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

    # 4. Predict with hybrid model, output is a dictionary:
    # {"xgb_chunk_proba", "cnn_chunk_proba",
    # "fused_chunk_proba", "final_proba", "final_prediction"}
    predictions = predict(
        xgb_model=xgb_model,
        cnn_model=cnn_model,
        xgb_features=xgb_df,
        cnn_features=cnn_df,
    )

    final = result["final_prediction"]

    return {
        "prediction": final["predicted_label"],
        "confidence": final["confidence"],
        "class_probabilities": final["class_probabilities"],
        "cycles_analysed": len(mel_spec_list)
    }
