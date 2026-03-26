import os
import io
import librosa
import pickle
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from smart_stethoscope.interface.main import preprocess_for_prediction, predict

CNN_MODEL_PATH = os.getenv("MODEL_PATH", "gs://smart-stethoscope/cnn_model.keras")
XGB_MODEL_PATH = os.getenv("MODEL_PATH", "gs://smart-stethoscope/xgb_model.pkl")
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
with open(XGB_MODEL_PATH, "rb") as f:
    xgb_model = pickle.load(f)

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
    }


@app.post("/predict")
async def predict_audio(
    audio_file: UploadFile = File(...), start: float = Form(...), end: float = Form(...)
):
    # 1. Read audio file into memory as bytes
    audio_bytes = await audio_file.read()

    # 2. Load audio into numpy array — no disk write needed
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # 3. Preprocess: resample, slice cycles, trim for both xbg and cnn
    xgb_df, cnn_df = preprocess_for_prediction(audio, sr, start, end)

    # 4. Predict with hybrid model, output is a dictionary:
    # {"xgb_chunk_proba", "cnn_chunk_proba",
    # "fused_chunk_proba", "final_proba", "final_prediction"}
    predictions = predict(
        xgb_model=xgb_model,
        cnn_model=cnn_model,
        xgb_features=xgb_df,
        cnn_features=cnn_df,
    )

    # 5. Output
    return {
        # 🎯 Final decision
        "prediction": predictions["final_prediction"],
        "final_proba": (
            predictions["final_proba"].tolist()
            if hasattr(predictions["final_proba"], "tolist")
            else predictions["final_proba"]
        ),
    }
