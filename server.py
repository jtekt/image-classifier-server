'''
Author: Maxime MOREILLON
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

app = Flask(__name__)
CORS(app)

MODEL_FOLDER_NAME = "model"
model_loaded = False
try:
    model = keras.models.load_model(MODEL_FOLDER_NAME)
    model_loaded = True
except Exception as e:
    pass

# Environment variables
IMAGE_WIDTH = os.getenv("IMAGE_WIDTH")
IMAGE_HEIGHT = os.getenv("IMAGE_HEIGHT")
MODEL_VERSION = os.getenv("MODEL_VERSION") or 1


@app.route('/', methods=['GET'])
def home():

    return json.dumps( {
    'applicationname': 'Image classifier server',
    'author': 'Maxime MOREILLON',
    'version': '1.0.0',
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
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

        # Image Preprocessing
        if IMAGE_WIDTH and IMAGE_HEIGHT:
            image_numpy = cv2.resize(image_numpy, dsize=(int(IMAGE_WIDTH), int(IMAGE_HEIGHT)), interpolation=cv2.INTER_CUBIC)

        image_numpy = image_numpy/255.00

        # Append to list
        image_list.append(image_numpy)

    # convert list to np array
    model_input = np.array(image_list)

    inference_start = time.time()

    # Prediction
    model_output = model(model_input)

    inference_time = time.time() - inference_start

    # converting output to numpy array
    model_output_numpy = model_output.numpy()

    # Preparing response as JSON
    response = json.dumps( {
    'predictions': model_output_numpy.tolist(),
    'model_version': MODEL_VERSION,
    'inference_time': inference_time,
    } )

    # Sending response
    return response


if __name__ == '__main__':
    app.run('0.0.0.0',7436)
