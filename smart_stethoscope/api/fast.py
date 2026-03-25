import os
import io
import numpy as np
import pandas as pd
import librosa
from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
from smart_stethoscope.ml_logic.preprocessing import audio_preprocessing

# ================================
# Model loading
# ================================
MODEL_PATH = os.getenv("MODEL_PATH", "gs://smart-stethoscope/best_cnn_model.keras")
model = keras.models.load_model(MODEL_PATH)

DISEASE_MAPPING_INV = {
    0: 'Healthy', 1: 'COPD', 2: 'URTI',
    3: 'Bronchiectasis', 4: 'Pneumonia', 5: 'Bronchiolitis'
}

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
        "model_loaded": model is not None
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

    # 4. Preprocess each cycle and collect mel spectrograms
    mel_specs = []
    for _, row in annotations.iterrows():
        _, mel_spec = audio_preprocessing(audio, sr, row["start"], row["end"])
        if mel_spec.shape[0] > 0:
            mel_specs.append(mel_spec)

    # 5. Stack all cycles into a single array for CNN input
    features = np.concatenate(mel_specs, axis=0)

    # 6. Predict per cycle — CNN returns probabilities, argmax gives class
    probabilities = model.predict(features)
    predicted_ints = np.argmax(probabilities, axis=1)

    # 7. Majority vote across cycles → single prediction per recording
    prediction_int = int(np.bincount(predicted_ints).argmax())

    return {
        "prediction": DISEASE_MAPPING_INV[prediction_int],
        "cycles_analysed": len(predicted_ints),
        "cycle_predictions": [DISEASE_MAPPING_INV[i] for i in predicted_ints.tolist()]
    }
