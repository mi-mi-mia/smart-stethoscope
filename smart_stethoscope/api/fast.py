import os
import io
import numpy as np
import pandas as pd
import librosa
from fastapi import FastAPI, UploadFile, File, Form
import pickle
from tensorflow import keras

from smart_stethoscope.ml_logic.preprocessing import (
    preprocess_audio,
    build_mel_spectrogram_dataset,
)
from smart_stethoscope.interface.main import preprocess_for_prediction, predict
from smart_stethoscope.ml_logic.model import predict_hybrid
from smart_stethoscope.params import TARGET_SAMPLING_RATE

# Load once at startup.
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_cnn_model.keras")
xgb_model = pickle.load(open("models/xgb_model.pkl","rb")) #PLACEHOLDER
cnn_model = keras.models.load_model(MODEL_PATH) #PLACEHOLDER



DISEASE_MAPPING_INV = {

    0: "Healthy",
    1: "COPD",
    2: "URTI",
    3: "Bronchiectasis",
    4: "Pneumonia",
    5: "Bronchiolitis",
}

app = FastAPI()


@app.get("/")
def index():
    return {"status": "API is online"}


@app.post("/predict")
async def predict_audio(
    audio_file: UploadFile = File(...),
    start: float=Form(...) ,
    end: float=Form(...)
):
    # 1. Read audio file into memory as bytes
    audio_bytes = await audio_file.read()

    # 2. Load audio into numpy array — no disk write needed
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # 3. Preprocess: resample, slice cycles, trim for both xbg and cnn
    xgb_df, cnn_df = preprocess_for_prediction(audio, sr, start, end)

    # 5. Predict with hybrid model, output is a dictionary:
    # {"xgb_chunk_proba", "cnn_chunk_proba",
    # "fused_chunk_proba", "final_proba", "final_prediction"}
    predictions = predict(xgb_model=xgb_model, cnn_model=cnn_model,
                   xgb_df=xgb_df, cnn_array = cnn_df
                   )

    xgb_chunk_proba = predictions["xgb_chunk_proba"]
    cnn_chunk_proba = predictions["cnn_chunk_proba"]
    fused_chunk_proba = predictions["fused_chunk_proba"]
    final_proba = predictions["final_proba"]
    final_prediction = int(predictions["final_prediction"])
    # 6. Output
    return {
        # 🎯 Final decision
        "prediction": DISEASE_MAPPING_INV[final_prediction],
        "final_prediction_int": final_prediction,
        "final_proba": final_proba.tolist() if hasattr(final_proba, "tolist") else final_proba,

        # 📊 Model outputs (useful for debugging / UI)
        "xgb_chunk_proba": np.array(xgb_chunk_proba).tolist(),
        "cnn_chunk_proba": np.array(cnn_chunk_proba).tolist(),
        "fused_chunk_proba": fused_chunk_proba.tolist(),
    }
