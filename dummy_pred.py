import numpy as np

async def pred_result(image_list):
    image_len = len(image_list)
    
    result = []
    rng = np.random.default_rng()
    for i in range(image_len):
        random_result = rng.random()
        result.append(random_result)
        
    return result