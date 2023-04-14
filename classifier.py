import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from fastapi import HTTPException
from os import getenv, path
from dotenv import load_dotenv
from time import time
import json
import io
import mlflow 
from config import mlflow_tracking_uri

load_dotenv()

class Classifier:

    def __init__(self):
        self.model_path = "./model"
        self.model_loaded = False

        # Attribute to hold additional information regarding the model
        self.model_info = {  }

        self.mlflow_model = None

        # Setting model parameters using env
        if getenv('CLASS_NAMES'):
            print('Classes set from env')
            self.model_info['classe_names'] = getenv("CLASS_NAMES").split(',')

        # Setting model parameters using env
        if mlflow_tracking_uri and getenv('MLFLOW_MODEL_VERSION') and getenv('MLFLOW_MODEL_NAME'):
            self.mlflow_model = {
                "model": getenv('MLFLOW_MODEL_NAME'),
                "version": getenv('MLFLOW_MODEL_VERSION')
            }

        try:
            self.load_model()
        except Exception as e:
            print('[AI] Failed to load model')
            print(e)

    def readModelInfo(self):
        file_path = path.join(self.model_path, 'modelInfo.json')
        with open(file_path, 'r') as openfile:
            return json.load(openfile)

    def load_model(self):

        # Reset model info
        self.model_info = {}

        print('[AI] Loading model')
        if self.mlflow_model:
            model_name = self.mlflow_model["model"]
            model_version = self.mlflow_model["version"]

            print(f'[AI] Downloading model {model_name} v{model_version} from MLflow at {mlflow_tracking_uri}')
            self.model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
            self.model_info['mlflow_url'] = f'{mlflow_tracking_uri}/#/models/{model_name}/versions/{model_version}'
            self.model_loaded = True

        else :
            print(f'[AI] Loading from local directory at {self.model_path}')
            self.model = keras.models.load_model(self.model_path)
            self.model_loaded = True

            # Get model info from .json file
            try:
                jsonModelInfo = self.readModelInfo()
                self.model_info = {**self.model_info, **jsonModelInfo}
            except:
                print('Failed to load .json model information')

        print('[AI] Model loaded')

        

    async def load_image_from_request(self, file):
        fileBuffer = io.BytesIO(file)

        target_size = None

        if hasattr(self.model, 'input'):
            target_size = (self.model.input.shape[1] , self.model.input.shape[2])

        elif hasattr(self.model, 'metadata'):
            input_shape = self.model.metadata.signature.inputs.to_dict()[0]['tensor-spec']['shape']
            target_size = (input_shape[1],input_shape[2])


        img = keras.preprocessing.image.load_img( fileBuffer, target_size=target_size)
        img_array = keras.preprocessing.image.img_to_array(img)

        # Create batch axis
        return tf.expand_dims(img_array, 0).numpy()  


    def get_class_name(self, prediction):
        # Name output if possible
        max_index = np.argmax(prediction)
        return self.model_info['class_names'][max_index]

    async def predict(self, file):

        if not self.model_loaded:
            raise HTTPException(status_code=503, detail='Model not loaded')

        inference_start_time = time()

        model_input = await self.load_image_from_request(file)
        model_output = self.model.predict(model_input)
        prediction = model_output[0]

        inference_time = time() - inference_start_time

        response = {
            'prediction': prediction.tolist(),
            'inference_time': inference_time
        }

        # Add class name if class names available
        if 'class_names' in self.model_info:
            response['class_name'] = self.get_class_name(prediction)

        print(f'[AI] Prediction: {prediction}')

        return response
