import requests
import base64

IMAGE_PATH = './example.jpg'
URL = 'http://localhost:7071/predict'

files = { 'images' : [base64.b64encode(open(IMAGE_PATH,'rb').read()).decode('utf-8')] }

# Send the image
print(f'Infering from to {URL}')
response = requests.post(URL, json=files)

# Check if inference is successful
if response.status_code ==  200:
    print(response.json())        
else:
    print(f'Inference failed with code {response.status_code}')
    print(response.text)
