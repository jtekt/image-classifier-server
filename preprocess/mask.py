### SACサンプルコード


import cv2
import os
import numpy as np
import json

import time
import base64


# output_dir = ""


async def process(base64,params):
    all_split_imgs = []
    radius = 322
    
    print(params)
    for idx, result1 in enumerate(base64):
        
        
        split_imgs_list = []
        print(result1.shape)

        new_center = (335, 335)

        mask1 = np.zeros_like(result1)
        cv2.circle(mask1, new_center, radius, (255, 255, 255), thickness=-1)
        result1 = cv2.bitwise_and(result1, mask1)

        alpha_channel = np.ones_like(result1[:, :, 0]) * 255
        cv2.circle(alpha_channel, new_center, radius, 0, thickness=-1)

        print(result1.shape)
        mask2 = np.ones((670, 670, 3), dtype=np.float32) * 255
            
        print(mask2.shape)
        cv2.circle(mask2, new_center, 225, (0, 0, 0), thickness=-1)
        result2 = cv2.bitwise_and(result1, mask2)
            # cv2.circle(result2, (335, 335), 5, (0, 255, 255), -1)
            # center_file = os.path.join(center_path, f'1_{image_file}')
            # cv2.imwrite(center_file, result2)
        result2[result2[:, :, 3] == 0] = [0, 0, 0, 255]
        result2 = result2[:, :, :3]

        print("トリミング後サイズ", result2.shape)
        
        split_imgs_list.append(result2)
        split_imgs = np.array(split_imgs_list)
        all_split_imgs.append(split_imgs)
        
        
        # output_file = os.path.join(output_dir, f"split_image_{idx}.jpg")
        # cv2.imwrite(output_file, result2)
        # print(f"Saved: {output_file}")
    
    all_split_imgs = np.concatenate(all_split_imgs, axis=0)        
    return all_split_imgs