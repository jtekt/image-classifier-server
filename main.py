from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from classifier import Classifier
from utils import getGpus, lookDeeperIfNeeded, load_image_from_request, base64_to_image_list
import zipfile
import io
from os import makedirs
import sys
from config import prevent_model_update, mlflow_tracking_uri
from pydantic import BaseModel
import shutil
from typing import List, Optional, Tuple
from PIL import Image
import base64
import numpy as np

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
    name: str
    version: str

@app.get("/")
async def root():
    response = {
    "application_name": "image classifier server",
    "author": "Maxime MOREILLON, Shion ITO",
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
async def predict(request: Request):
    content_type = request.headers.get("content-type").split(";", 1)[0].strip().lower()

    if content_type == "multipart/form-data":
        form = await request.form()
        img_list = []

        for key, val in form.items():
            if key.startswith("image"):
                img_array = load_image_from_request(await val.read())
                img_list.append(img_array)
        img_list = np.stack(img_list, axis=0)

    elif content_type == "application/json":
        payload = await request.json()
        img_list = await base64_to_image_list(payload["images"])
    
    else:
        return error(400, 'content type not supported')

    result = await classifier.predict(img_list)

    return result

@app.post("/model")
async def upload_model(model: UploadFile = File(...)):
    
    if prevent_model_update:
        raise HTTPException(status_code=403, detail="Model update is forbidden")
    
    # save model file according to file extension
    if model.filename.endswith('.zip'):
        # reset model folder
        shutil.rmtree("./model", ignore_errors=True)
        makedirs("./model", exist_ok=True)
        with io.BytesIO(await model.read()) as tmp_stream, zipfile.ZipFile(tmp_stream, 'r') as zip_ref:
            zip_ref.extractall("./model")
        # unify folder structure when unzipping
        lookDeeperIfNeeded('./model')
    elif model.filename.endswith('.onnx'):
        # reset model folder
        shutil.rmtree("./model", ignore_errors=True)
        makedirs("./model", exist_ok=True)
        file_path = f'./model/{model.filename}'
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(model.file, buffer)
    else:
        # error script
        raise HTTPException(status_code=400, detail="Invalid file type. Only .zip or .onnx files are accepted.")
    
    # load model
    classifier.load_model_from_local()
    
    return classifier.model_info["type"]

# Proxying the MLflow REST API for the classifier server GUI
# TODO: Put those in a dedicated route

if mlflow_tracking_uri:
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    @app.get("/mlflow/models")
    async def getMlflowModels(search: str='', page_token: str=""):
        models = []
        
        filter_string = f"name ILIKE '%{search}%'"
        res = client.search_registered_models(filter_string=filter_string, page_token=page_token)

        for model in res:
            models.append(model)

        return {"models": models, "page_token": res.token}
    
    @app.get("/mlflow/models/{model}/versions")
    async def getMlflowModelVersions(model):
        versions = []
        for version in client.search_model_versions(f"name='{model}'"):
            versions.append(version)
        return versions
    
    @app.put("/model")
    async def updateMlflowModel(mlflowModel: MlflowModel):
        if prevent_model_update:
            raise HTTPException(status_code=403, detail="Model update is forbidden")
        classifier.load_model_from_mlflow(mlflowModel.dict()["name"], mlflowModel.dict()["version"])
        return {
            "result": "OK"
        }