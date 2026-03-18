
### fourth version

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# # Allow all requests (optional, good for development purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )
# need to include python-multipart in the requirements.txt
@app.get("/")
def index():
    return {"status": "API is online", "version": "0.1-minimal"}
@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    # not yet sure what this does
    _ = await file.read()
    # Hardcoded repsonse (Mnimal API)
    return { "filename": file.filename,
            "prediction": "Healthy",
            "confidence": 0.98,
            "features_extracted": { "centroid": 1500.5, "bandwidth": 2200.1, "rolloff": 4500.0, "mfcc_1": -250.3 } }
