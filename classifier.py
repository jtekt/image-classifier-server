from tensorflow import keras
import numpy as np
import cv2
from os import getenv
from dotenv import load_dotenv
from time import time


load_dotenv()

class Classifier:

    def __init__(self):
        self.model_path = "model"
        self.image_width = getenv("IMAGE_WIDTH")
        self.image_height = getenv("IMAGE_HEIGHT")
        self.load_model()

    def load_model(self):
        print('[AI] Loading model...')
        self.model = keras.models.load_model(self.model_path)
        print('[AI] Model loaded')


    async def load_image_from_request(self, file):
         img_data = await file.read()
         nparr = np.frombuffer(img_data, np.uint8)
         return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def image_prepropcessing(self, image):
        # Resizing
        target_size = (int(self.image_width), int(self.image_height))
        image_resized = cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_CUBIC)

        return image_resized

    async def predict(self, file):

        inference_start_time = time()


        img = await self.load_image_from_request(file)
        img_processed = self.image_prepropcessing(img)
        input = np.array([img_processed])
        output_tensor = self.model(input)
        output = output_tensor.numpy().tolist()[0]

        inference_time = time() - inference_start_time


        return {
        'prediction': output,
        'inference_time': inference_time,
        }



classifier = Classifier()
