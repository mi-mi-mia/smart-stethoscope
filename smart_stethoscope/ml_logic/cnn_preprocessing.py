# ==================================================
# KEIRA DRAFT FOR CNN PREPROCESSING
# TO BE MOVED INTO MAIN PREPROCESSING FILE
# ==================================================
# Full Preprocessing outputs xgb_df and cnn_array
# Assumes XGBoost preprocessing already captured:
# - load, trim + normalise from audio function
# - xgb feature extraction function
# Same chunks to be used for XGB and CNN


import librosa as lb
import numpy as np

TARGET_SR = 12000
CHUNK_DURATION = 6
SAMPLES_PER_CHUNK = TARGET_SR * CHUNK_DURATION

STEP = TARGET_SR * 2  # 2s overlap (safe default)


def generate_chunks(y):
    # for xgb and cnn (use same chunks)
    """
    Splits audio into overlapping chunks
    """
    chunks = [
        y[i:i + SAMPLES_PER_CHUNK]
        for i in range(0, len(y), STEP)
        if len(y[i:i + SAMPLES_PER_CHUNK]) == SAMPLES_PER_CHUNK
    ]
    return chunks


def chunk_to_mel(chunk):
    mel = lb.feature.melspectrogram(
        y=chunk,
        sr=12000,
        n_mels=128
    )
    mel_db = lb.power_to_db(mel, ref=np.max)
    return mel_db[..., np.newaxis].astype(np.float32)


def build_cnn_array(chunks):
    """
    chunks: list of audio chunks (already created upstream)

    returns:
        np.array (n_chunks, 128, time, 1) --> cnn input
    """
    mel_list = [chunk_to_mel(chunk) for chunk in chunks]
    return np.stack(mel_list)
