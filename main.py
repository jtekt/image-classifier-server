import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from classifier import Classifier
from utils import getGpus, lookDeeperIfNeeded
import zipfile
import io
from os import path, remove, mkdir
import sys
from config import prevent_model_update, mlflow_tracking_uri
from pydantic import BaseModel
import requests
import shutil
import numpy as np
from PIL import Image
import image_preprocess_mask
import base64
import base64_to_image
import dummy_pred
import time



classifier = Classifier()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)




class getitem(BaseModel):
    images: list
    model: str
    preprocess: list
    preprocess_params: dict

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
    'model_info': {**classifier.model_infos},
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

import os
from fastapi.responses import JSONResponse

@app.post("/test")

async def test(item:getitem):

    return item.model



@app.post("/predict")
# async def predict(image: bytes = File()):
async def predict(item:getitem):
    import importlib
    
    result = []

    preprocess_params = item.preprocess_params
    model_name = item.model
    process_list = item.preprocess
    images_list = item.images
    output_folder = "/home/digital-orin02/python/request_api/image_after"

    # preprocessディレクトリのパス
    preprocess_dir = os.path.join(os.path.dirname(__file__), 'preprocess')
    sys.path.append(preprocess_dir)
    
    # preprocessディレクトリ内のすべてのPythonファイルをインポート
    module_names = [f[:-3] for f in os.listdir(preprocess_dir) if f.endswith('.py')]


    modules = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        modules[module_name] = module

    start_time = time.time()  # 計測開始
    process_image_list = await base64_to_image.base64_to_image(item.images)
    end_time = time.time()  # 計測終了
    elapsed_time = end_time - start_time
    print("base64_to_image_time",elapsed_time)
    
    start_time = time.time()  # 計測開始
    # print(process_image_list)
    for i,process in enumerate(process_list):
        # print(process_image_list.shape)
        print(f"----------------------------{process_list[i]}_process-----------------------------")
        
        
        if process in preprocess_params:
            params = preprocess_params[process]
            for param_key, param_value in params.items():
                print(f"{param_key}: {param_value}")
        
        if process == 'split' :
            process_image_list,process_info = await modules[process_list[i]].process(process_image_list,params)
        else:
            process_image_list = await modules[process_list[i]].process(process_image_list,params)
    

    start_time = time.time()  # 計測開始
    result = await classifier.predict_batch(process_image_list, item.model,process_info)
    end_time = time.time()  # 計測終了
    elapsed_time = end_time - start_time

    print("predict_time", elapsed_time)

    return result


@app.post("/model")
async def upload_model(model: UploadFile = File(...)):

    if prevent_model_update:
        raise HTTPException(status_code=403, detail="Model update is forbidden")
    
    # save model file according to file extension
    if model.filename.endswith('.zip'):
        print("zip")
        # reset model folder
        shutil.rmtree("./model")
        mkdir("./model")
        with io.BytesIO(await model.read()) as tmp_stream, zipfile.ZipFile(tmp_stream, 'r') as zip_ref:
            zip_ref.extractall("./model")
        # unify folder structure when unzipping
        lookDeeperIfNeeded('./model')
    elif model.filename.endswith('.onnx'):
        print("onnx")
        # reset model folder
        shutil.rmtree("./model")
        mkdir("./model")
        file_path = f'./model/{model.filename}'
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(model.file, buffer)
    else:
        # error script
        raise HTTPException(status_code=400, detail="Invalid file type. Only .zip and .onnx files are accepted.")
    
    # load model
    classifier.load_model_from_local()
    
    return classifier.model_info["type"]





# Proxying the MLflow REST API for the classifier server GUI
# TODO: Put those in a dedicated route

# if mlflow_tracking_uri:
#     import mlflow
#     from mlflow import MlflowClient
#     mlflow.set_tracking_uri(mlflow_tracking_uri)
#     client = MlflowClient()



#     @app.get("/mlflow/models")
#     async def getMlflowModels():
#         models = []
#         for model in client.search_registered_models():
#             models.append(model)
#         return models
    
#     @app.get("/mlflow/models/{model}/versions")
#     async def getMlflowModelVersions(model):
#         versions = []
#         for version in client.search_model_versions(f"name='{model}'"):
#             versions.append(version)
#         return versions
    
#     @app.put("/model")
#     async def updateMlflowModel(mlflowModel: MlflowModel):
#         if prevent_model_update:
#             raise HTTPException(status_code=403, detail="Model update is forbidden")
#         classifier.load_model_from_mlflow(mlflowModel.dict()["name"], mlflowModel.dict()["version"])
#         return {
#             "result": "OK"
#         }
        
        