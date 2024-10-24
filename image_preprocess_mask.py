# import
import cv2
import os
import numpy as np
import json

import time
import base64


# trimming
# trmi_start_x = int(os.getenv("TRIM_START_X") or 210)
# trmi_start_y = int(os.getenv("TRIM_START_Y") or 240)
# trmi_start_x = int(os.getenv("TRIM_START_X") or 299)
# trmi_start_y = int(os.getenv("TRIM_START_Y") or 172)
trmi_start_x = int(299)
trmi_start_y = int(172)

# splited image size
# w = int(os.getenv("SPLIT_IMG_W") or 180)
# h = int(os.getenv("SPLIT_IMG_H") or 180)
w = 670
h = 670

# resize_img
# w_resize = int(os.getenv("IMG_RESIZE_W") or 224)
# h_resize = int(os.getenv("IMG_RESIZE_H") or 224)
# resize_img
w_resize = int(224)
h_resize = int(224)

split_start_points = json.loads(os.getenv("SPLIT_START_POINT") or "[[0, 0]]")
batch_size = len(split_start_points)

# # split start
# split_start_points = json.loads(os.getenv("SPLIT_START_POINT") or "[[0, 0], [0, 90]]")
# batch_size = len(split_start_points)

center_x = 634
center_y = 507
radius = 322


output_dir = "/home/digital-orin02/python/request_api/image_process"



async def split_img_with_cv2(images_base64):
    
    for idx, img_base64 in enumerate(images_base64):
        # Base64デコード
        img_data = base64.b64decode(img_base64)
        np_img = np.frombuffer(img_data, np.uint8)
        print(idx)
        
        # 画像読み込み
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        split_imgs = np.empty(
            (len(split_start_points), h_resize, w_resize, 3), dtype=np.float32
        )
        for i, start_point in enumerate(split_start_points):
            y = trmi_start_y + start_point[1]
            x = trmi_start_x + start_point[0]
            split_img = img[y : y + h, x : x + w]
            # split_img = msk(split_img)
            split_img = cv2.resize(split_img, dsize=(h_resize, w_resize))
            split_img = cv2.cvtColor(split_img, cv2.COLOR_BGR2RGB)
            split_imgs[i] = split_img


        
        output_file = os.path.join(output_dir, f"split_image_{idx}.jpg")
        cv2.imwrite(output_file, split_img)
        print(f"Saved: {output_file}")
    print(split_imgs) 
    return split_imgs

