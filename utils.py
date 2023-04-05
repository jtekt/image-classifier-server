import tensorflow as tf

def gpuAvailable():
  print('Checking if GPU is available')
  devices = tf.config.list_physical_devices('GPU')
  print(devices)
