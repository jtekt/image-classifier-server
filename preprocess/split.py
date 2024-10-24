# import cv2
# import os
# import numpy as np
# import json

# import time
# import base64



# output_dir = "/home/digital-orin02/python/request_api/image_process"


# async def process(image, Params): 

#     if image is None or image.size == 0:
#         raise ValueError("Image is empty or not loaded correctly.")



#     print(image)
#     W = Params["W"]
#     H = Params["H"]
#     X = Params["X"]
#     Y = Params["Y"]
#     w = Params["w"]
#     h = Params["h"]
#     cols = Params["cols"]
#     rows = Params["rows"]
#     index = 0
#     image_info = []
#     all_split_imgs = []
#     all_split_data = {}  
    
#     for idx, image in enumerate(image):
#         output = []
#         for row in range(rows):
#             for col in range(cols):
                
#                 index = index + 1

#                 print("index",index)
#                 dx = (W - cols * w) / (cols - 1) if cols > 1 else 0
#                 dy = (H - rows * h) / (rows - 1) if rows > 1 else 0

#                 x = round(X + col * (w + dx))
#                 y = round(Y + row * (h + dy))
                
#                 print("dx,dy = ",dx,dy)
#                 print("x,y = ",x,y)
                
#                 image_info.append({
#                         'index' : index,
#                         'start_x': x,
#                         'start_y': y,
#                         'width': w,
#                         'height': h
#                 }) 
#                 cropped = image[y:y + h, x:x + w]
#                 print(cropped.shape)
#                 cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
#                 output.append(cropped)
#         all_split_imgs.append(output)
#     all_split_imgs = np.concatenate(all_split_imgs, axis=0)
        
#     return all_split_imgs,image_info


import cv2
import os
import numpy as np
import json

import time
import base64


output_dir = "/home/digital-orin02/python/request_api/image_process"


async def process(image, Params): 

    if image is None or image.size == 0:
        raise ValueError("Image is empty or not loaded correctly.")


    print(image)
    W = Params["W"]
    H = Params["H"]
    X = Params["X"]
    Y = Params["Y"]
    w = Params["w"]
    h = Params["h"]
    cols = Params["cols"]
    rows = Params["rows"]
    index = 0
    image_info = []
    all_split_imgs = []
    all_split_data = {}  
    
    for idx, image in enumerate(image):
        output = []
        for row in range(rows):
            for col in range(cols):
                
                index = index + 1

                print("index", index)
                dx = (W - cols * w) / (cols - 1) if cols > 1 else 0
                dy = (H - rows * h) / (rows - 1) if rows > 1 else 0

                x = round(X + col * (w + dx))
                y = round(Y + row * (h + dy))
                
                print("dx,dy = ", dx, dy)
                print("x,y = ", x, y)
                
                image_info.append({
                        'index': index,
                        'start_x': x,
                        'start_y': y,
                        'width': w,
                        'height': h,
                        'row': row + 1,  # Adding row information
                        'col': col + 1   # Adding column information
                }) 
                cropped = image[y:y + h, x:x + w]
                print(cropped.shape)
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                output.append(cropped)
        all_split_imgs.append(output)
    
    all_split_imgs = np.concatenate(all_split_imgs, axis=0)
        
    return all_split_imgs, image_info
