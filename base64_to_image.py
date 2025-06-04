import base64
import numpy as np
import cv2


def b64_to_np(img_base64):
    img_data = base64.b64decode(img_base64)

    np_img = np.frombuffer(img_data, np.uint8)

    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)


async def base64_to_image(images_base64):

    np_img_list = [b64_to_np(img_base64) for img_base64 in images_base64]
    return np_img_list