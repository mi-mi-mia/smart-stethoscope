import os
import io
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from tensorflow import keras

from smart_stethoscope.interface.main import preprocess_for_prediction, predict

# Load once at startup
CNN_MODEL_PATH = os.getenv("MODEL_PATH", "gs://smart-stethoscope/cnn_model.keras")
XGB_MODEL_PATH = os.getenv("MODEL_PATH", "gs://smart-stethoscope/xgb_model.keras")
cnn_model = keras.models.load_model(CNN_MODEL_PATH)
xgb_model = keras.models.load_model(XGB_MODEL_PATH)


app = FastAPI()


@app.get("/")
def index():
    return {"status": "API is online"}


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
        xgb_model=xgb_model, cnn_model=cnn_model, xgb_df=xgb_df, cnn_array=cnn_df
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
