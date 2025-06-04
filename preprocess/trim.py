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

center_x = 640
center_y = 486
radius = 322


output_dir = "/home/digital-orin02/python/request_api/image_process"

async def process(np_img, params):
    all_split_imgs = []
    all_split_data = {}  
    
    print(params)
    for idx, img in enumerate(np_img):
        split_imgs_list = []
        split_coords_list = []  
        
        for i, start_point in enumerate(split_start_points):
            y = trmi_start_y + start_point[1]
            x = trmi_start_x + start_point[0]
            split_img = img[y : y + h, x : x + w]
            # split_img = cv2.resize(split_img, dsize=(h_resize, w_resize))
            split_img = cv2.cvtColor(split_img, cv2.COLOR_BGR2RGB)
            split_imgs_list.append(split_img)
            
            # 辞書に座標を保存
            split_coords_list.append({
                "index": i,           # 画像のインデックス
                "x": x,               # トリミング開始X座標
                "y": y,               # トリミング開始Y座標
                "width": w,           # トリミング領域の幅
                "height": h           # トリミング領域の高さ
            })
            print(split_img.shape)
        
        split_imgs = np.array(split_imgs_list)
        all_split_imgs.append(split_imgs)
        

        all_split_data[f"split_image_{idx}"] = split_coords_list

        output_file = os.path.join(output_dir, f"split_image_{idx}.jpg")
        cv2.imwrite(output_file, split_img)
        print(f"Saved: {output_file}")

    all_split_imgs = np.concatenate(all_split_imgs, axis=0)
    

    return all_split_imgs, all_split_data
