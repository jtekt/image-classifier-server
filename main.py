from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from classifier import Classifier
from utils import getGpus
import zipfile
import io
import sys
from config import model_name, model_version, mlflow_tracking_uri

classifier = Classifier()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
    "application_name": "image classifier server",
    "author": "Maxime MOREILLON",
    "version": "0.4.2",
    "model_loaded": classifier.model_loaded,
    'model_info': {**classifier.model_info},
    'gpu': len(getGpus()),
    "mlflow_tracking_uri": mlflow_tracking_uri,
    'versions': {
        'python': sys.version,
        'tensorflow': tf.__version__
        }
    }

@app.post("/predict")
async def predict(image: bytes = File()):
    result = await classifier.predict(image)
    return result

@app.post("/model")
async def upload_model(model: bytes = File()):
    if mlflow_tracking_uri and model_name and model_version:
        raise HTTPException(status_code=409, detail="Cannot POST models when MLflow used as origin")
        
    fileBuffer = io.BytesIO(model)
    with zipfile.ZipFile(fileBuffer) as zip_ref:
        zip_ref.extractall('./model')
        
    classifier.load_model()
    return {"file_size": len(model)}