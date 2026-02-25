from time import time
import io
import tempfile
import shutil
import pathlib
import json

import tensorflow as tf
import numpy as np
import yaml
import onnxruntime
import mlflow

from config import (
    mlflow_tracking_uri, provider, warm_up, warm_up_batch_size,
    class_names, mlflow_model_name, mlflow_model_version,
    PROV_TRT,
    PROV_VINO,
)
import mlflow_patchcore

if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

class Classifier:

    def __init__(self):
        self.model_path = pathlib.Path("./model")
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
        if list(self.model_path.glob("*")):
            self.load_model_from_local()
        # load model from mlflow
        elif mlflow_tracking_uri and mlflow_model_name and mlflow_model_version:
            try:
                self.load_model_from_mlflow(mlflow_model_name, mlflow_model_version)
            except Exception as e:
                print('[AI] Failed to load model')
                print(e)

    def readModelInfo(self):
        file_path = str(self.model_path / "modelInfo.json")
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
            shutil.rmtree(self.model_path, ignore_errors=True)
            self.model_path.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as dname:
                mlflow.artifacts.download_artifacts(
                    artifact_uri=model_uri,
                    dst_path=dname,
                )
                for p in pathlib.Path(dname).glob("*"):
                    shutil.move(str(p), self.model_path)

            self.load_model_from_local()

            self.model_info['mlflow_url'] = f'{mlflow_tracking_uri}/#/models/{model_name}/versions/{model_version}'
            self.model_info['origin'] = 'mlflow'
            return
        elif mlmodel['flavors'].get('mlflow_patchcore'):
            print('[AI] Loading patchcore_cvj model')
            self.model_info['type'] = 'patchcore_cvj'
            self.model_info['patchcore_cvj_threshold'] = mlmodel["metadata"].get('patchcore_threshold', 0.0)
            self.model_info['patchcore_cvj_class_index'] = mlmodel["metadata"].get('patchcore_class_index', 0)
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
            fais_nn = list(self.model_path.glob("fais_nn"))
            onnx    = list(self.model_path.glob("*.onnx"))
            if (fais_nn and onnx):
                self.load_model_from_patchcore_cvj()
            elif onnx:
                self.model_name = onnx[0].name
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
        print(f'[AI] Loading from local directory at {str(self.model_path)}')

        self.model = tf.keras.models.load_model(str(self.model_path))

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
        print(f'[AI] Loading from local directory at {str(self.model_path)}')

        file_path = self.model_path / self.model_name
        if not file_path.is_file():
            raise ValueError(f"Model file {str(file_path)} does not exist")

        # Set provider of onnxruntime
        available_providers = onnxruntime.get_available_providers()

        if provider in available_providers:
            providers = [provider]
        else:
            providers = available_providers

        if PROV_TRT in providers:
            providers[providers.index(PROV_TRT)] = (PROV_TRT, {"trt_fp16_enable": True})
        if PROV_VINO in providers:
            if tuple([int(v) for v in onnxruntime.__version__.split(".")]) > (1, 17, 3):
                providers[providers.index(PROV_VINO)] = (PROV_VINO, {"device_type": "GPU"})
            else:
                providers[providers.index(PROV_VINO)] = (PROV_VINO, {"device_type": "GPU_FP32"})

        self.model = onnxruntime.InferenceSession(str(file_path), providers=providers)

        self.model_loaded = True
        self.model_info['origin'] = "folder"
        self.model_info['type'] = "onnx"
        self.model_info['providers'] = providers

        print('[AI] Model loaded')
        print(f'[AI] ONNX Runtime Providers: {str(providers)}')

    def load_model_from_patchcore_cvj(self):

        self.model_info = {}
        print('[AI] Loading patchcore_cvj model')
        print(f'[AI] Loading from local directory at {str(self.model_path)}')

        self.model = mlflow_patchcore._load_pyfunc(str(self.model_path / "model.onnx"))
        self.model_info['patchcore_cvj_threshold'] = self.model.patchcore_threshold
        self.model_info['patchcore_cvj_class_index'] = self.model.patchcore_index

        self.model_loaded = True
        self.model_info['origin'] = "folder"
        self.model_info['type'] = "patchcore_cvj"

        print('[AI] Model loaded')

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


    def resize_image(self, img_array):

        if self.model_info['format'] == 'NCHW':
            img_array = tf.image.resize(img_array, self.target_size[1:3], method="bilinear").numpy()
            img_array = img_array.transpose((0, 3, 1, 2)) / 255.0
        else:
            img_array = tf.image.resize(img_array, self.target_size[0:2], method="bilinear").numpy()

        return img_array

    def get_class_name(self, prediction):
        # Name output if possible
        max_index = np.argmax(prediction)
        return self.model_info['class_names'][max_index]

    def warm_up(self):
        initial_startup_time_start = time()
        # make dummy data
        model_input = np.zeros((warm_up_batch_size, *self.target_size), dtype='float32')
        # predict
        for i in range(5):
            res = self.predict(model_input)
            print(f'[AI] warm up: {res["inference_time"]:5.3}')
        # Separate by type of output
        initial_startup_time = time() - initial_startup_time_start
        print('[AI] The initial startup of model is done.')
        print('[AI] Initial startup time:', initial_startup_time, 's')
        return

    def predict(self, file):

        inference_start_time = time()

        if not self.model_loaded:
            raise Exception("No loaded model")

        model_input = self.resize_image(file)

        # Separate by existing functions
        if hasattr(self.model, 'predict'):
            model_output = self.model.predict(model_input)
        elif hasattr(self.model, 'run'):
            output_names = [outp.name for outp in self.model.get_outputs()]
            input = self.model.get_inputs()[0]
            model_output = self.model.run(output_names, {input.name: model_input})

        # Separate by type of output
        if isinstance(model_output, (dict, list, tuple)):
            if isinstance(model_output, dict):
                # dict -> list
                model_output = [m for m in model_output.values()]

            if model_input.shape[0] == 1:
                model_output = [m[0] for m in model_output]

            # (list, tuple) -> ndarray
            prediction = model_output[0]
        else:
            if model_input.shape[0] == 1:
                prediction = model_output[0]
            else:
                prediction = model_output

        if prediction.ndim >= 3:
            prediction_list = []
            for i in range(len(prediction)):
                pred = prediction[i].max()
                prediction_list.append(pred)
            prediction = np.array(prediction_list)

        inference_time = time() - inference_start_time

        response = {
            'prediction': prediction.tolist(),
            'inference_time': inference_time
        }

        if isinstance(model_output, (list, tuple)):
            if self.model_info['type'] == 'patchcore_cvj':
                # patchcore model
                response['patchcore_cvj_raw'] = model_output[1].tolist()
                response['patchcore_cvj_normalized'] = model_output[2].tolist()

        # Add class name if class names available
        if 'class_names' in self.model_info:
            response['class_name'] = self.get_class_name(prediction)

        return response
