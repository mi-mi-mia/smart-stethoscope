import os
import io
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
    client = storage.Client()
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    return joblib.load(buffer)

def load_json_from_gcs(gcs_path: str):
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
