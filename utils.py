from tensorflow.python.client import device_lib

def getGpus():
  devices = device_lib.list_local_devices()
  gpus = [d for d in devices if d.device_type == "GPU"]
  return gpus
  

