import requests

IMAGE_PATH = './example.jpg'
URL = 'http://localhost:7071/predict'

files = { 'image' : ('myImage.jpg', open(IMAGE_PATH,'rb').read()) }

# Send the image
print(f'Infering from to {URL}')
response = requests.post(URL, files=files)

# Check if inference is successful
if response.status_code ==  200:
    print(response.json())        
else:
    print(f'Inference failed with code {response.status_code}')
    print(response.text)
