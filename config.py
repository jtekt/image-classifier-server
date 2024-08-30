from os import environ
from dotenv import load_dotenv

load_dotenv()

prevent_model_update = environ.get('PREVENT_MODEL_UPDATE')
mlflow_tracking_uri = environ.get('MLFLOW_TRACKING_URI')
provider = environ.get('ONNXRUNTIME_PROVIDERS')
warm_up = environ.get('WARM_UP')