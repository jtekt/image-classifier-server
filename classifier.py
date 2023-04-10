import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from os import getenv, path
from dotenv import load_dotenv
from time import time
import json
import io
from config import model_name, model_version, mlflow_tracking_uri
import mlflow 

load_dotenv()

class Classifier:

    def __init__(self):
        self.model_path = "./model"
        self.model_loaded = False

        # Attribute to hold additional information regarding the model
        self.model_info = {  }

        # Setting model parameters using env
        if getenv('CLASS_NAMES'):
            print('Classes set from env')
            self.model_info['classe_names'] = getenv("CLASS_NAMES").split(',')

        self.load_model()

    def readModelInfo(self):
        file_path = path.join(self.model_path, 'modelInfo.json')
        with open(file_path, 'r') as openfile:
            return json.load(openfile)

    def load_model(self):
        # TODO: Throw an error if the model cannot be loaded
        # Note: This will make the container crash if function called outside of FastAPI

        if mlflow_tracking_uri and model_name and model_version:
            print(f'[AI] Downloading model {model_name} v{model_version} from MLflow at {mlflow_tracking_uri}')
            self.model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
            self.model_info['mlflow_url'] = f'{mlflow_tracking_uri}/#/models/{model_name}/versions/{model_version}'
            self.model_loaded = True

        else :
            try:
                print('[AI] Loading model...')
                self.model = keras.models.load_model(self.model_path)
                print('[AI] Model loaded')
                self.model_loaded = True

            except Exception as e:
                print('[AI] Failed to load model')
                print(e)
                self.model_loaded = False

            # Get model info from .json file
            try:
                jsonModelInfo = self.readModelInfo()
                self.model_info = {**self.model_info, **jsonModelInfo}
                
            except:
                print('[AI] Failed to load .json model information')


    async def load_image_from_request(self, file):
        fileBuffer = io.BytesIO(file)

        target_size = None

        if mlflow_tracking_uri and model_name and model_version:
            # Getting input shape from MLflow
            input_shape = self.model.metadata.signature.inputs.to_dict()[0]['tensor-spec']['shape']
            target_size = (input_shape[1],input_shape[2])

        else:
            target_size = (model.input.shape[1] , model.input.shape[2])

        img = keras.preprocessing.image.load_img( fileBuffer, target_size=target_size)
        img_array = keras.preprocessing.image.img_to_array(img)

        # Create batch axis
        return tf.expand_dims(img_array, 0).numpy()  


    def get_class_name(self, prediction):
        # Name output if possible
        max_index = np.argmax(prediction)
        return self.model_info['class_names'][max_index]

    async def predict(self, file):

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
