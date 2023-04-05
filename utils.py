from tensorflow.python.client import device_lib

def gpuAvailable():
  print('Checking if GPU is available')
  devices = device_lib.list_local_devices()
  print(devices)
