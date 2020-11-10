'''
Image classifier server
Maxime MOREILLON
'''

from flask import Flask, escape, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import time


load_dotenv()

# Environment variables
IMAGE_WIDTH = os.getenv("IMAGE_WIDTH")
IMAGE_HEIGHT = os.getenv("IMAGE_HEIGHT")
MODEL_VERSION = os.getenv("MODEL_VERSION") or 'none'
MODEL_NAME = os.getenv("MODEL_NAME") or 'none'

normalize = True
if os.getenv("NORMALIZE"):
    if os.getenv("NORMALIZE").lower() == 'false':
        normalize = False



# Loading AI model
MODEL_FOLDER_NAME = "model"
model_loaded = False
try:
    model = keras.models.load_model(MODEL_FOLDER_NAME)
    model_loaded = True
    print('AI model loaded')
except Exception as e:
    print('Failed to load the AI model')

# Flask app
app = Flask(__name__)
CORS(app)

def preprocess_image(image):
    # Resizing
    if IMAGE_WIDTH and IMAGE_HEIGHT:
        image = cv2.resize(image, dsize=(int(IMAGE_WIDTH), int(IMAGE_HEIGHT)), interpolation=cv2.INTER_CUBIC)

    # Normalization
    if normalize:
        image = image/255.00

    return image

@app.route('/', methods=['GET'])
def home():

    return jsonify( {
    'application_name': 'Image classifier server',
    'author': 'Maxime MOREILLON',
    'version': '1.0.4',
    'model_name': MODEL_NAME,
    'model_version': MODEL_VERSION,
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'Normalization': normalize,
    'model_loaded': model_loaded,
    } )


@app.route('/predict', methods=['POST'])
def predict():

    # Check if the request contains files
    if not request.files:
        print('Files not found in request body')
        return 'Files not found in request body', 400

    # Initialize an empty image list
    image_list = []

    for key, file in request.files.items():
        image_numpy = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        image_preprocessed = preprocess_image(image_numpy)
        image_list.append(image_preprocessed)

    # convert list to np array
    model_input = np.array(image_list)

    inference_start_time = time.time()

    # AI Prediction
    try:
        model_output = model(model_input)
    except Exception as e:
        print('AI prediction failed: {}'.format(e))
        return 'AI prediction failed: {}'.format(e), 500

    inference_time = time.time() - inference_start_time

    # converting output to numpy array
    model_output_numpy = model_output.numpy()

    print('AI predictions: {}'.format(model_output_numpy.tolist()))

    # Sending response
    return jsonify( {
    'predictions': model_output_numpy.tolist(),
    'inference_time': inference_time,
    'model_version': MODEL_VERSION,
    'model_name': MODEL_NAME,
    } )


if __name__ == '__main__':
    app.run('0.0.0.0',7436)
