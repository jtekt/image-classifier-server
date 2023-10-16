from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from classifier import Classifier
from utils import getGpus
import zipfile
import io
from os import path
import sys
from config import prevent_model_update, mlflow_tracking_uri
from pydantic import BaseModel
import requests
from pydantic import BaseModel

classifier = Classifier()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MlflowModel(BaseModel):
    model: str
    version: str

@app.get("/")
async def root():
    response = {
    "application_name": "image classifier server",
    "author": "Maxime MOREILLON",
    "version": "0.7.0",
    "model_loaded": classifier.model_loaded,
    'model_info': {**classifier.model_info},
    "mlflow_tracking_uri": mlflow_tracking_uri,
    'gpu': len(getGpus()),
    'update_allowed': not prevent_model_update,
    'versions': {
        'python': sys.version,
        'tensorflow': tf.__version__
        }
    }
    if classifier.mlflow_model:
        response["mlflow_model"] = {**classifier.mlflow_model}
    return response

@app.post("/predict")
async def predict(image: bytes = File()):
    result = await classifier.predict(image)
    return result

@app.post("/model")
async def upload_model(model: bytes = File()):
    if prevent_model_update:
        raise HTTPException(status_code=403, detail="Model update is forbidden")
    
    model_name = None
    
    fileBuffer = io.BytesIO(model)
    with zipfile.ZipFile(fileBuffer) as zip_ref:
        zip_ref.extractall('./model')
        names = zip_ref.namelist()
        filename = path.dirname(names[0])
    
    for name in names:
        base, ext = path.splitext(name)
        if ext == '.onnx':
            model_name = name
            
    if model_name:
        classifier.model_name = model_name
        classifier.load_model_from_onnx()
    else:
        classifier.model_name = filename
        classifier.load_model_from_keras()
    
    return classifier.model_info["load_model"]

# Proxying the MLflow REST API for the classifier server GUI
# TODO: Put those in a dedicated route

if mlflow_tracking_uri:
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    @app.get("/mlflow/models")
    async def getMlflowModels():
        models = []
        for model in client.search_registered_models():
            models.append(model)
        return models
    
    @app.get("/mlflow/models/{model}/versions")
    async def getMlflowModelVersions(model):
        versions = []
        for version in client.search_model_versions(f"name='{model}'"):
            versions.append(version)
        return versions
    
    @app.put("/mlflow")
    async def updateMlflowModel(mlflowModel: MlflowModel):
        if prevent_model_update:
            raise HTTPException(status_code=403, detail="Model update is forbidden")
        classifier.load_model_from_mlflow(mlflowModel.dict()["model"], mlflowModel.dict()["version"])
        return {
            "OK"
        }