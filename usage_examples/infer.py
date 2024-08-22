import requests

IMAGE_PATH = './example.jpg'
URL = 'http://localhost:7071/predict'

files = { 'image' : ('myImage.jpg', open(IMAGE_PATH,'rb').read()) }

# If you want to use heatmap, change the param to 'true'
params = {'heatmap': 'false'}

# Send the image
print(f'Infering from to {URL}')
response = requests.post(URL, params=params, files=files)

# Check if inference is successful
if response.status_code ==  200:
    
    if response.headers['Content-Type'] == 'application/json':
        print(response.json())
        with open('./result.json', "wb") as f:
            f.write(response.content)
    elif response.headers['Content-Type'] == 'image/png':
        with open('./result.png', "wb") as f:
            f.write(response.content)
        
else:
    print(f'Inference failed with code {response.status_code}')
    print(response.text)
