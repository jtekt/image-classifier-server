import base64
import numpy as np
import cv2

def b64_to_np(img_base64):
    img_data = base64.b64decode(img_base64)
    # 画像データに変換
    np_img = np.frombuffer(img_data, np.uint8)
    # OpenCVを使って画像としてデコード
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)

async def base64_to_image(images_base64):
    # np_img_list = []
    
    # for idx, img_base64 in enumerate(images_base64):
    #     # Base64デコード
    #     print("**********************************",idx)
    #     img_data = base64.b64decode(img_base64)
    #     # 画像データに変換
    #     np_img = np.frombuffer(img_data, np.uint8)
    #     # OpenCVを使って画像としてデコード
    #     img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    #     # np_img_list.append(img)
    #     np_img_list.append(img.astype(np.float32))
    
    # # すべての画像をスタックして1つのNumPy配列にする
    # final_array = np.stack(np_img_list)
    
    np_img_list=[b64_to_np(img_base64) for img_base64 in images_base64]
    final_array=np.array(np_img_list)
    
    print("final_array shape:", final_array.shape)    
    return final_array
