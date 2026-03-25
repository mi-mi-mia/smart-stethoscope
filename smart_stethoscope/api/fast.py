import os
import io
import numpy as np
import pandas as pd
import librosa
from fastapi import FastAPI, UploadFile, File
import pickle
from tensorflow import keras

from smart_stethoscope.ml_logic.preprocessing import (
    preprocess_audio,
    build_mel_spectrogram_dataset,
)
from smart_stethoscope.interface.main import preprocess_for_prediction
from smart_stethoscope.params import TARGET_SAMPLING_RATE

# Load once at startup.
# this still loads locally? needs to be changed to gcloud?
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_cnn_model.keras") #needs to be updated?
xgb_model = pickle.load(open("models/xgb_model.pkl","rb")) #will this be a pickkle file?
cnn_model = keras.models.load_model(MODEL_PATH)

#CODE FROM MLOPS exc for loading model from gcs
#def load_model():
#      client = storage.Client()
#       blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
#
#       try:
#           latest_blob = max(blobs, key=lambda x: x.updated)
#           latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
#           latest_blob.download_to_filename(latest_model_path_to_save)
#
#           latest_model = keras.models.load_model(latest_model_path_to_save)
#
#           print("✅ Latest model downloaded from cloud storage")
#
#           return latest_model
#       except:
#           print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")
#
#           return None

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
    #annotation_file: UploadFile = File(...)
    start: float,
    end: float,
    audio_file: UploadFile = File(...), annotation_file: UploadFile = File(...)
):
    # 1. Read both files into memory as bytes
    audio_bytes = await audio_file.read()
    #annotation_bytes = await annotation_file.read()

    # 2. Load audio into numpy array — no disk write needed
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # 3. Parse annotation .txt into a DataFrame
    #annotations = pd.read_csv(
    #    io.StringIO(annotation_bytes.decode("utf-8")),
    #    sep="\t",
    #    names=["start", "end", "crackles", "wheezes"]
    #)

    # 4. Preprocess: resample, slice cycles, trim
    #preprocess_audio(audio, sr, start, end)
    xgb_df, cnn_df = preprocess_for_prediction(audio, sr, start, end)
    # 5. Convert breathing cycles into mel spectrograms for CNN
    #features = build_mel_spectrogram_dataset(padded_audios)

    # 6. Predict per cycle — CNN returns probabilities, argmax gives class
    probabilities_xgb = xgb_model.predict_proba(xgb_df) # shape: (n_cycles, 6)
    probablities_cnn = cnn_model.predict(cnn_df)

    probabilities = model.predict(features)  # shape: (n_cycles, 6)
    predicted_ints = np.argmax(probabilities, axis=1)  # one per cycle

    # 7. Majority vote across cycles → single prediction per recording
    prediction_int = int(np.bincount(predicted_ints).argmax())

    return {
        "prediction": DISEASE_MAPPING_INV[prediction_int],
        "cycles_analysed": len(predicted_ints),
        "cycle_predictions": [DISEASE_MAPPING_INV[i] for i in predicted_ints.tolist()],
    }
