from tensorflow import keras
import numpy as np
import cv2
from os import getenv, path
from dotenv import load_dotenv
from time import time
import json

load_dotenv()

class Classifier:

    def __init__(self):
        self.model_path = "./model"
        self.resize = {
            'width': getenv("RESIZE_WIDTH"),
            'height': getenv("RESIZE_HEIGHT"),
        }
        self.classes = None

        if getenv("CLASSES"):
            self.classes = getenv("CLASSES").split(',')

        self.model_loaded = False
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
            modelInfo = self.readModelInfo()
            if 'class_names' in modelInfo:
                self.classes = modelInfo['class_names']
        except:
            print('[AI] Failed to load model information')

    async def load_image_from_request(self, file):
         img_data = await file.read()
         nparr = np.frombuffer(img_data, np.uint8)
         decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
         #cv2.imwrite('image.jpg', decoded_image)
         return decoded_image

    def image_prepropcessing(self, image):
        # Resizing if needed
        if self.resize['width'] is None or self.resize['height'] is None:
            print('[Preprocessing] Skipping resize')
            return image

        target_size = (int(self.resize['width']), int(self.resize['height']))
        image_resized = cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_CUBIC)

        return image_resized

    def class_naming(self, output):
        # Name output if possible

        if self.classes is None:
            return output

        max_index = output.index(max(output))
        name = self.classes[max_index]

        return name

    async def predict(self, file):

        inference_start_time = time()


        img = await self.load_image_from_request(file)
        img_processed = self.image_prepropcessing(img)
        input = np.array([img_processed])
        output_tensor = self.model(input)
        output = output_tensor.numpy().tolist()[0]

        inference_time = time() - inference_start_time

        output_named = self.class_naming(output)

        print(f'[AI] Prediction: {output_named}')


        return {
        'prediction': output_named,
        'inference_time': inference_time,
        }
