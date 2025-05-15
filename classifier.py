from time import time
import io

import tensorflow as tf
import numpy as np
import yaml

from os import path
import onnxruntime
from time import time
import io
from glob import glob
import json
import mlflow

from config import (
    mlflow_tracking_uri, provider, warm_up,
    class_names, mlflow_model_name, mlflow_model_version,
)


if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

class Classifier:
    
    def __init__(self):
        self.model_path = "./model"
        self.model_loaded = False

        # Attribute to hold additional information regarding the model
        self.model_info = {  }

        self.mlflow_model = None

        # Setting model parameters using env
        if class_names:
            print('Classes set from env')
            self.model_info['class_names'] = class_names.split(',')
        
        # load model files first if they exist in the local directory
        # load model from local directory
        if glob(path.join(self.model_path, "*")):
            self.load_model_from_local()
        # load model from mlflow
        elif mlflow_tracking_uri and mlflow_model_name and mlflow_model_version:
            try:
                self.load_model_from_mlflow(mlflow_model_name, mlflow_model_version)
            except Exception as e:
                print('[AI] Failed to load model')
                print(e)

    def readModelInfo(self):
        file_path = path.join(self.model_path, 'modelInfo.json')
        with open(file_path, 'r') as openfile:
            return json.load(openfile)
        

    def load_model_from_mlflow(self, model_name, model_version):
        # load any format model mlflow 
        # Reset model info
        self.model_info = {}
        if hasattr(self, 'model'):
            del self.model
        
        print(f'[AI] Downloading model {model_name} v{model_version} from MLflow at {mlflow_tracking_uri}')
        
        model_uri = f'models:/{model_name}/{model_version}'
        
        mlmodel = yaml.safe_load(mlflow.artifacts.load_text(f'{model_uri}/MLmodel'))

        if mlmodel['flavors'].get('tensorflow'):
            print('[AI] Loading keras model')
            self.model_info['type'] = 'keras'
        elif mlmodel['flavors'].get('onnx'):
            print('[AI] Loading onnx model')
            self.model_info['type'] = 'onnx'
        else:
            print('[AI] Loading model')
            self.model_info['type'] = 'other'

        self.model = mlflow.pyfunc.load_model(model_uri)
        self.model_info['mlflow_url'] = f'{mlflow_tracking_uri}/#/models/{model_name}/versions/{model_version}'
        self.model_loaded = True
        self.model_info['origin'] = 'mlflow'
    
        print('[AI] Model loaded')

        self.get_target_size()
        
        if warm_up:
            self.warm_up()
    
    def load_model_from_local(self):
        # load model from local directory
        # load ONNX files first, if available
        if hasattr(self, 'model'):
            del self.model
        try:
            if glob(path.join(self.model_path, "*.onnx")):
                self.model_name = path.basename(glob(path.join(self.model_path, "*.onnx"))[0])
                self.load_model_from_onnx()
            else:
                self.load_model_from_keras()
       
        except Exception as e:
            print('[AI] Failed to load model from local directory')
            print(e)

        self.get_target_size()
            
        if warm_up:
            self.warm_up()
    
    def load_model_from_keras(self):

        # Reset model info
        self.model_info = {}

        print('[AI] Loading keras model')
        print(f'[AI] Loading from local directory at {self.model_path}')
        
        self.model = tf.keras.models.load_model(self.model_path)
        
        self.model_loaded = True
        self.model_info['origin'] = "folder"
        self.model_info['type'] = "keras"

        # Get model info from .json file
        try:
            jsonModelInfo = self.readModelInfo()
            self.model_info = {**self.model_info, **jsonModelInfo}
        except:
            print('Failed to load .json model information')

        print('[AI] Model loaded')
        
    def load_model_from_onnx(self):
        
        self.model_info = {}
        print('[AI] Loading onnx model')
        print(f'[AI] Loading from local directory at {self.model_path}')

        file_path = path.join(self.model_path, self.model_name)
        if not path.isfile(file_path):
            raise ValueError(f"Model file {file_path} does not exist")
        
        # Set provider of onnxruntime
        available_providers = onnxruntime.get_available_providers()
            
        if provider in available_providers:
            providers = [provider]
        else:
            providers = available_providers
            
        self.model = onnxruntime.InferenceSession(file_path, providers=providers)
        
        self.model_loaded = True
        self.model_info['origin'] = "folder"
        self.model_info['type'] = "onnx"
        self.model_info['providers'] = providers

        print('[AI] Model loaded')
        print(f'[AI] ONNX Runtime Providers: {str(providers)}')
        
    def get_target_size(self):
        # Separate by the method of getting input size
        if hasattr(self.model, 'input'):
            self.target_size = self.model.input.shape[1:4].as_list()

        elif hasattr(self.model, 'metadata'):
            input_shape = self.model.metadata.signature.inputs.to_dict()[0]['tensor-spec']['shape']
            self.target_size = input_shape[1:4]
            
        elif hasattr(self.model, 'get_inputs'):
            input_shape = self.model.get_inputs()[0].shape
            self.target_size = input_shape[1:4]

        if self.target_size.index(min(self.target_size)) == 0:
            print('[AI] This model is channels first.')
            self.model_info['format'] = 'NCHW'
        elif self.target_size.index(min(self.target_size)) == 2:
            print('[AI] This model is channels last.')
            self.model_info['format'] = 'NHWC'
        else:
            print('[AI] This model is from other.')
            self.model_info['format'] = 'other'

        
    async def load_image_from_request(self, file):
        fileBuffer = io.BytesIO(file)

        img = tf.keras.preprocessing.image.load_img(fileBuffer)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        if self.model_info['format'] == 'NCHW':
            img_array = tf.image.resize(img_array, self.target_size[1:3], method="bilinear").numpy()
            img_array = img_array.transpose((2, 0, 1)) / 255.0
        else:
            img_array = tf.image.resize(img_array, self.target_size[0:2], method="bilinear").numpy()

        # Create batch axis
        return np.expand_dims(img_array, axis=0)

    def get_class_name(self, prediction):
        # Name output if possible
        max_index = np.argmax(prediction)
        return self.model_info['class_names'][max_index]
    
    def warm_up(self):
        initial_startup_time_start = time()
        # make dummy data
        model_input = np.zeros((1, *self.target_size), dtype='float32')
        # predict
        if hasattr(self.model, 'predict'):
            _ = self.model.predict(model_input)
        elif hasattr(self.model, 'run'):
            output_names = [outp.name for outp in self.model.get_outputs()]
            input = self.model.get_inputs()[0]
            _ = self.model.run(output_names, {input.name: model_input})[0]
        # Separate by type of output
        initial_startup_time = time() - initial_startup_time_start
        print('[AI] The initial startup of model is done.')
        print('[AI] Initial startup time:', initial_startup_time, 's')
        return
    
    async def predict(self, file):
        
        inference_start_time = time()
        
        model_input = await self.load_image_from_request(file)
        
        # Separate by existing functions
        if hasattr(self.model, 'predict'):
            model_output = self.model.predict(model_input)
        elif hasattr(self.model, 'run'):
            output_names = [outp.name for outp in self.model.get_outputs()]
            input = self.model.get_inputs()[0]
            model_output = self.model.run(output_names, {input.name: model_input})[0]
        
        # Separate by type of output
        if isinstance(model_output, dict):
            prediction = model_output['pred'][0]
        else:
            prediction = model_output[0]

        if prediction.ndim != 1:
            prediction = np.array(prediction.max())

        inference_time = time() - inference_start_time

        response = {
            'prediction': prediction.tolist(),
            'inference_time': inference_time
        }

        # Add class name if class names available
        if 'class_names' in self.model_info:
            response['class_name'] = self.get_class_name(prediction)

        return response
