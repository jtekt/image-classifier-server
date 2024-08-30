import tensorflow as tf
from tensorflow import keras
import numpy as np
from fastapi import Response
from os import getenv, path
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
import matplotlib.pyplot as plt

load_dotenv()

if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

class Classifier:
    
    def __init__(self):
        self.model_path = "./model"
        self.model_loaded = False
        self.last_conv_layer_name = 'top_conv'

        # Attribute to hold additional information regarding the model
        self.model_info = {  }

        self.mlflow_model = None

        # Setting model parameters using env
        if getenv('CLASS_NAMES'):
            print('Classes set from env')
            self.model_info['class_names'] = getenv("CLASS_NAMES").split(',')
        
        # load model files first if they exist in the local directory
        # load model from local directory
        if glob(path.join(self.model_path, "*")):
            self.load_model_from_local()
        # load model from mlflow
        elif mlflow_tracking_uri and getenv('MLFLOW_MODEL_VERSION') and getenv('MLFLOW_MODEL_NAME'):
            try:
                self.load_model_from_mlflow(getenv('MLFLOW_MODEL_NAME'), getenv('MLFLOW_MODEL_VERSION'))
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
        
        mlmodel_fname = mlflow.models.model.MLMODEL_FILE_NAME
        repo = mlflow.store.artifact.artifact_repository_registry.get_artifact_repository(model_uri)
        repo._download_file(mlmodel_fname, mlmodel_fname)
        mlmodel = mlflow.models.Model.load(mlmodel_fname)

        if mlmodel.flavors.get('tensorflow'):
            print('[AI] Loading keras model')
            tmp_model = mlflow.keras.load_model(model_uri)
            
            self.model = tf.keras.models.Model(
                tmp_model.input,
                [tmp_model.get_layer(self.last_conv_layer_name).output, tmp_model.output]
            )
            
            del tmp_model

            self.model_info['mlflow_url'] = f'{mlflow_tracking_uri}/#/models/{model_name}/versions/{model_version}'
            self.model_loaded = True
            self.model_info['origin'] = "mlflow"
            self.model_info['type'] = 'keras'
            
        elif mlmodel.flavors.get('onnx'):
            print('[AI] Loading onnx model')
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.model_info['mlflow_url'] = f'{mlflow_tracking_uri}/#/models/{model_name}/versions/{model_version}'
            self.model_loaded = True
            self.model_info['origin'] = "mlflow"
            self.model_info['type'] = 'onnx'
            
        else:
            print('[AI] Loading model')
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.model_info['mlflow_url'] = f'{mlflow_tracking_uri}/#/models/{model_name}/versions/{model_version}'
            self.model_loaded = True
            self.model_info['origin'] = "mlflow"
            self.model_info['type'] = 'other'
    
        print('[AI] Model loaded')
        
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
            
        if warm_up:
            self.warm_up()
    
    def load_model_from_keras(self):

        # Reset model info
        self.model_info = {}

        print('[AI] Loading keras model')
        
        print(f'[AI] Loading from local directory at {self.model_path}')
        
        tmp_model = keras.models.load_model(self.model_path)
        
        self.model = tf.keras.models.Model(
            tmp_model.inputs,
            [tmp_model.get_layer(self.last_conv_layer_name).output, tmp_model.output]
        )
        
        del tmp_model
        
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
            self.target_size = (self.model.input.shape[1] , self.model.input.shape[2])

        elif hasattr(self.model, 'metadata'):
            input_shape = self.model.metadata.signature.inputs.to_dict()[0]['tensor-spec']['shape']
            self.target_size = (input_shape[1], input_shape[2])
            
        elif hasattr(self.model, 'get_inputs'):
            input_shape = self.model.get_inputs()[0].shape
            self.target_size = (input_shape[1], input_shape[2])
        
    async def load_image_from_request(self, file):
        fileBuffer = io.BytesIO(file)
        image = Image.open(io.BytesIO(file))
        
        self.target_size = None
        
        self.get_target_size()

        img = keras.preprocessing.image.load_img(fileBuffer, target_size=self.target_size)
        img_array = keras.preprocessing.image.img_to_array(img)

        # Create batch axis
        return tf.expand_dims(img_array, 0).numpy(), image

    def get_class_name(self, prediction):
        # Name output if possible
        max_index = np.argmax(prediction)
        return self.model_info['class_names'][max_index]
    
    def warm_up(self):
        # make dummy data
        self.get_target_size()
        input_ = np.ones(self.target_size, dtype='int8')
        num_pil = Image.fromarray(input_)
        num_byteio = io.BytesIO()
        num_pil.save(num_byteio, format='png')
        num_bytes = num_byteio.getvalue()
        
        initial_startup_time_start = time()
        # reshape dummy data
        fileBuffer = io.BytesIO(num_bytes)
        img = keras.preprocessing.image.load_img(fileBuffer, target_size=self.target_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        # Create batch axis
        model_input = tf.expand_dims(img_array, 0).numpy()
        # predict
        if self.model_info['type'] == 'keras':
            __, model_output = self.model(model_input, training=False)
            model_output = np.array(model_output)
        else:
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
        initial_startup_time = time() - initial_startup_time_start
        print('[AI] The initial startup of model is done.')
        print('[AI] Initial startup time:', initial_startup_time, 's')
    
    def makeGradcamHeatmap(self, model_input, image, pred_index=None, alpha=0.3):
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.model(model_input, training=False)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = np.int8(255 * heatmap.numpy())
        jet = plt.get_cmap('jet')
        
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize(image.size)
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        img = np.asarray(image)
        
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        return superimposed_img, preds
    
    async def predict(self, file, heatmap):
        if heatmap:
            response = await self.makeHeatmap(file)
        else:
            response = await self.makePredictJson(file)
        
        return response
    
    async def makeHeatmap(self, file):
        model_input, image = await self.load_image_from_request(file)
        
        superimposed_img, __ = self.makeGradcamHeatmap(model_input, image)
        
        img_bytes = io.BytesIO()
        superimposed_img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        return Response(content=img_bytes, media_type='image/png')
    
    async def makePredictJson(self, file):
        
        inference_start_time = time()
        
        model_input, _ = await self.load_image_from_request(file)
        
        # Separate by existing functions
        if self.model_info['type'] == 'keras':
            __, model_output = self.model(model_input, training=False)
            model_output = np.array(model_output)
        else:
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
