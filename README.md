# Image classifier server

This is an image classificatier server which can take any exported Keras AI model and expose it through an HTTP API.

<p align="center">
  <img src="./docs/image_classifier_server.gif">
</p>

## API

| Endpoint | Method | body/query | Description
| --- | --- | --- | --- |
| /predict | POST | Image as 'image' field of multipart/form-data | Get model inference on the image |
| /model | POST | .zip of model export as 'model' field of multipart/form-data | Upload a new model |



## Usage examples

### Using dedicated Dockerfile
```
FROM moreillon/image-classifier-server-fastapi
COPY path-to-your-model ./model
```

### Running in a development environment

```
uvicorn main:app --reload --port 7071 --host 0.0.0.0

```