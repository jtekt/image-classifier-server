from os import environ, path
from dotenv import load_dotenv

load_dotenv()


model_name = environ.get('MODEL_NAME')
model_version = environ.get('MODEL_VERSION')
mlflow_tracking_uri = environ.get('MLFLOW_TRACKING_URI')

