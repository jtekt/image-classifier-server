import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from os import getenv, path
from dotenv import load_dotenv
from time import time
import json
import io

load_dotenv()

class Classifier:

    def __init__(self):
        self.model_path = "./model"
        self.model_loaded = False

        # Attribute to hold additional information regarding the model
        self.model_info = {  }

        if getenv('CLASSES'):
            self.model_info['classe_names'] = getenv("CLASSES").split(',')

        self.load_model()

    def readModelInfo(self):
        file_path = path.join(self.model_path, 'modelInfo.json')
        with open(file_path, 'r') as openfile:
            return json.load(openfile)

    def load_model(self):

        # The loading of the model itself
        try:
            print('[AI] Loading model...')
            self.model = keras.models.load_model(self.model_path)
            print('[AI] Model loaded')

            self.model_loaded = True
        except:
            print('[AI] Failed to load model')
            self.model_loaded = False

        # Trying to get model info from .json file
        # TODO: More than just classes
        try:
            jsonModelInfo = self.readModelInfo()
            self.model_info = {**self.model_info, **jsonModelInfo}
            
        except:
            print('[AI] Failed to load .json model information')


    async def load_image_from_request(self, file):
        fileBuffer = io.BytesIO(file)

        target_size = None
        if 'input_size' in self.model_info:
            # TODO: DOUBLE CHECK IF HEIGHT FIRST OR WIDTH FIRST
            target_size = (self.model_info['input_size']['height'], self.model_info['input_size']['width'])

        img = keras.preprocessing.image.load_img( fileBuffer, target_size=target_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        return tf.expand_dims(img_array, 0)  # Create batch axis


    def get_class_name(self, prediction):
        # Name output if possible

        max_index = np.argmax(prediction)
        name = self.model_info['class_names'][max_index]

        return name

    async def predict(self, file):

        inference_start_time = time()

        model_input = await self.load_image_from_request(file)

        model_output = self.model.predict(model_input)

        prediction = model_output[0]

        inference_time = time() - inference_start_time

        response = {
            'prediction': prediction.tolist(),
            'inference_time': inference_time
        }



        # Add class name if class names available
        if 'class_names' in self.model_info:
            print('MODEL INFO HAS CLASSES')
            response['class_name'] = self.get_class_name(prediction)

        print(f'[AI] Prediction: {prediction}')

        return response
