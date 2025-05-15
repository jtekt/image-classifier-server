from os import environ
from dotenv import load_dotenv

load_dotenv()

prevent_model_update = environ.get('PREVENT_MODEL_UPDATE')
mlflow_tracking_uri = environ.get('MLFLOW_TRACKING_URI')
provider = environ.get('ONNXRUNTIME_PROVIDERS')
warm_up = environ.get('WARM_UP')

class_names = environ.get('CLASS_NAMES')
mlflow_model_name = environ.get('MLFLOW_MODEL_NAME')
mlflow_model_version = environ.get('MLFLOW_MODEL_VERSION')
