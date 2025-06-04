import onnxruntime as ort

def check_onnx_cuda():
    providers = ort.get_available_providers()
    print("Available providers:", providers)
    
    if 'CUDAExecutionProvider' in providers:
        print("CUDA is available for ONNX Runtime.")
    else:
        print("CUDA is not available for ONNX Runtime.")

check_onnx_cuda()


import cv2

print(cv2.__version__)