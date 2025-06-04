import tensorflow as tf
from tensorflow import keras
import numpy as np
from fastapi import HTTPException
from os import getenv, path,listdir
import onnx
import onnxruntime
from dotenv import load_dotenv
from time import time
import json
import io
from glob import glob
import json
import mlflow 
from PIL import Image
from config import mlflow_tracking_uri, provider, warm_up
import traceback

load_dotenv()

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
        if getenv('CLASS_NAMES'):
            print('Classes set from env')
            self.model_info['class_names'] = getenv("CLASS_NAMES").split(',')
        
        # load model files first if they exist in the local directory
        # load model from local directory
        # if glob(path.join(self.model_path, "*")):
        #     self.load_model_from_local()
            
        self.models = {}
        self.load_models()
        
        # # load model from mlflow
        # elif mlflow_tracking_uri and getenv('MLFLOW_MODEL_VERSION') and getenv('MLFLOW_MODEL_NAME'):
        #     try:
        #         self.load_model_from_mlflow(getenv('MLFLOW_MODEL_NAME'), getenv('MLFLOW_MODEL_VERSION'))
        #     except Exception as e:
        #         print('[AI] Failed to load model')
        #         print(e)

    def load_models(self):
        subdirs = [d for d in listdir(self.model_path) if path.isdir(path.join(self.model_path, d))]
        for subdir in subdirs:
            model_dir = path.join(self.model_path, subdir)
            if glob(path.join(model_dir, "*")):
                try:
                    model = self.load_model_from_local(model_dir)
                    self.models[subdir] = model
                except Exception as e:
                    print(f'[AI] Failed to load model from {model_dir}')
                    print(e)
                    print(traceback.format_exc())
            elif mlflow_tracking_uri and getenv('MLFLOW_MODEL_VERSION') and getenv('MLFLOW_MODEL_NAME'):
                try:
                    self.load_model_from_mlflow(getenv('MLFLOW_MODEL_NAME'), getenv('MLFLOW_MODEL_VERSION'))
                except Exception as e:
                    print('[AI] Failed to load model from MLflow')
                    print(e)
        
        
    def readModelInfo(self):
        file_path = path.join(self.model_path, 'modelInfo.json')
        with open(file_path, 'r') as openfile:
            return json.load(openfile)
        

    def load_model_from_mlflow(self, model_name, model_version):
        # load any format model mlflow 
        # Reset model info
        self.model_info = {}
        
        print(f'[AI] Downloading model {model_name} v{model_version} from MLflow at {mlflow_tracking_uri}')

        self.model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        self.model_info['mlflow_url'] = f'{mlflow_tracking_uri}/#/models/{model_name}/versions/{model_version}'
        self.model_loaded = True
        self.model_info['origin'] = "mlflow"
        
        print('[AI] Model loaded')
        
        if warm_up:
            self.warm_up()
    
    def load_model_from_local(self,model_dir):
        # load model from local directory
        # load ONNX files first, if available
        model = None
        try:
            if glob(path.join(model_dir, "*.onnx")):
                self.model_name = path.basename(glob(path.join(model_dir, "*.onnx"))[0])
                model = self.load_model_from_onnx(model_dir)
            else:
                model = self.load_model_from_keras(model_dir)

            if warm_up:
                self.warm_up()
        except Exception as e:
            print('[AI] Failed to load model from local directory')
            print(e)
            print(traceback.format_exc())
        return model
        
        
        #--------------------------------------------------------------------------------------
        # load model from local directory
        # load ONNX files first, if available
        try:
            if glob(path.join(self.model_path, "*.onnx")):
                self.model_name = path.basename(glob(path.join(self.model_path, "*.onnx"))[0])
                self.load_model_from_onnx()
            else:
                self.load_model_from_keras()
                print("chcek point")

            if warm_up:
                self.warm_up()
        except Exception as e:
            print('[AI] Failed to load model from local directory')
            print(e)
            print(traceback.format_exc())
    
    def load_model_from_keras(self):

        print(self)
        # Reset model info
        self.model_info = {}
        
        print('[AI_keras] Loading model')
        
        print(f'[AI_keras] Loading from local directory at {self.model_path}')

    
        self.model = keras.models.load_model(self.model_path)
        self.model_loaded = True
        self.model_info['origin'] = "folder"
        self.model_info['type'] = "keras"

        print(self.model_info)
        # Get model info from .json file
        try:
            jsonModelInfo = self.readModelInfo()
            self.model_info = {**self.model_info, **jsonModelInfo}
        except:
            print('Failed to load .json model information')

        print('[AI] Model loaded')
        
    def load_model_from_onnx(self):
        
        self.model_info = {}
        print('[AI_onnx] Loading model')
        
        print(f'[AI_onnx] Loading from local directory at {self.model_path}')

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
            self.target_size = (self.model.input.shape[1] , self.model.input.shape[2])

        elif hasattr(self.model, 'metadata'):
            input_shape = self.model.metadata.signature.inputs.to_dict()[0]['tensor-spec']['shape']
            self.target_size = (input_shape[1], input_shape[2])
            
        elif hasattr(self.model, 'get_inputs'):
            input_shape = self.model.get_inputs()[0].shape
            self.target_size = (input_shape[1], input_shape[2])
        
    async def load_image_from_request(self, file):
        fileBuffer = io.BytesIO(file)

        self.target_size = None
        
        self.get_target_size()

        # img = keras.preprocessing.image.load_img(fileBuffer, target_size=self.target_size)

        img = Image.fromarray(file)
        img = img.convert("RGB")

        img_array = keras.preprocessing.image.img_to_array(img)

        # Create batch axis
        return tf.expand_dims(img_array, 0).numpy()

    def get_class_name(self, prediction):
        # Name output if possible
        max_index = np.argmax(prediction)
        return self.model_info['class_names'][max_index]
    
    # def warm_up(self):
    #     # make dummy data
    #     self.get_target_size()
    #     input_ = np.ones(self.target_size, dtype='int8')
    #     num_pil = Image.fromarray(input_)
    #     num_byteio = io.BytesIO()
    #     num_pil.save(num_byteio, format='png')
    #     num_bytes = num_byteio.getvalue()
        

    #     initial_startup_time_start = time()
    #     # reshape dummy data
    #     fileBuffer = io.BytesIO(num_bytes)
    #     img = keras.preprocessing.image.load_img(fileBuffer, target_size=self.target_size)
    #     img_array = keras.preprocessing.image.img_to_array(img)
    #     # Create batch axis
    #     model_input = tf.expand_dims(img_array, 0).numpy()
    #     # predict
    #     if hasattr(self.model, 'predict'):
    #         model_output = self.model.predict(model_input)
    #     elif hasattr(self.model, 'run'):
    #         output_names = [outp.name for outp in self.model.get_outputs()]
    #         input = self.model.get_inputs()[0]
    #         model_output = self.model.run(output_names, {input.name: model_input})[0]
    #     # Separate by type of output
    #     if isinstance(model_output, dict):
    #         prediction = model_output['pred'][0]
    #     else:
    #         prediction = model_output[0]
    #     initial_startup_time = time() - initial_startup_time_start
    #     print('[AI] The initial startup of model is done.')
    #     print('[AI] Initial startup time:', initial_startup_time, 's')
    
    
    
    async def predict_batch(self, image_list):

        predictions = []

        for img_data in image_list:

            print(img_data)
            prediction_result = await self.predict(img_data)
            
            predictions.append(prediction_result)  

        return predictions
    
        
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

        inference_time = time() - inference_start_time

        response = {
            'prediction': prediction.tolist(),
            'inference_time': inference_time
        }

        # Add class name if class names available
        if 'class_names' in self.model_info:
            response['class_name'] = self.get_class_name(prediction)

        return response
