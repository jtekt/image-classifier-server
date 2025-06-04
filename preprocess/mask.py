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


async def process(base64,params):
    all_split_imgs = []
    
    print(params)
    for idx, result1 in enumerate(base64):
        
        # split_imgs = np.empty(
        #     (len(split_start_points), h_resize, w_resize, 3), dtype=np.float32
        # )
        
        split_imgs_list = []
        print(result1.shape)

        new_center = (335, 335)
            # radiusの範囲を残す
        mask1 = np.zeros_like(result1)
        cv2.circle(mask1, new_center, radius, (255, 255, 255), thickness=-1)
        result1 = cv2.bitwise_and(result1, mask1)

        alpha_channel = np.ones_like(result1[:, :, 0]) * 255
        cv2.circle(alpha_channel, new_center, radius, 0, thickness=-1)
        # result1 = cv2.merge((result1, cv2.bitwise_not(alpha_channel)[:, :, None]))

        print(result1.shape)
            # 中心から内径の外側を残す
        mask2 = np.ones((670, 670, 3), dtype=np.float32) * 255
            
        print(mask2.shape)
        cv2.circle(mask2, new_center, 225, (0, 0, 0), thickness=-1)
            # cv2.circle(mask2, new_center, 225, (0, 0, 0), thickness=-1)
        result2 = cv2.bitwise_and(result1, mask2)

            # cv2.circle(result2, (335, 335), 5, (0, 255, 255), -1)
            # center_file = os.path.join(center_path, f'1_{image_file}')
            # cv2.imwrite(center_file, result2)
            # # 透過塗りつぶし
        result2[result2[:, :, 3] == 0] = [0, 0, 0, 255]
        result2 = result2[:, :, :3]

        print("トリミング後サイズ", result2.shape)
        
        split_imgs_list.append(result2)
        # split_imgs[i] = result2
        split_imgs = np.array(split_imgs_list)
        all_split_imgs.append(split_imgs)
        
        
        output_file = os.path.join(output_dir, f"split_image_{idx}.jpg")
        cv2.imwrite(output_file, result2)
        print(f"Saved: {output_file}")
    
    all_split_imgs = np.concatenate(all_split_imgs, axis=0)        
    return all_split_imgs