from os import environ, path
from dotenv import load_dotenv

load_dotenv()

prevent_model_update = environ.get('PREVENT_MODEL_UPDATE')
