# first version (from vlad for images)
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
#import cv2
import io
#from face_rec.face_detection import annotate_face

app = FastAPI()

# # Allow all requests (optional, good for development purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    ### Do cool stuff with your image.... For example face detection
    annotated_img = annotate_face(cv2_img)

    ### Encoding and responding with the image
    im = cv2.imencode('.png', annotated_img)[1] # extension depends on which format is sent from Streamlit
    return Response(content=im.tobytes(), media_type="image/png")

### second version

from typing import Annotated

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

#i think this turns the file into bytes which we dont need
@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

### third version

from fastapi import File, UploadFile, HTTPException

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        with open(file.filename, 'wb') as f:
            while contents := file.file.read(1024 * 1024):
                f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}
