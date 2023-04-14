from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from classifier import Classifier
from utils import getGpus
import zipfile
import io
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
        
    fileBuffer = io.BytesIO(model)
    with zipfile.ZipFile(fileBuffer) as zip_ref:
        zip_ref.extractall('./model')
    
    classifier.mlflow_model = None
    classifier.load_model()
    return {"file_size": len(model)}


@app.put("/model")
async def updateMlflowModel(mlflowModel: MlflowModel):
    if prevent_model_update:
        raise HTTPException(status_code=403, detail="Model update is forbidden")
    classifier.mlflow_model = mlflowModel.dict()
    classifier.load_model()
    return "OK"

# Proxying the MLflow REST API for the classifier server GUI
# TODO: Put those in a dedicated route

if mlflow_tracking_uri:

    from mlflow import MlflowClient
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