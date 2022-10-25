# Image classifier server

This is an image classificatier server which can take any exported Keras AI model and expose it through an HTTP API.

<p align="center">
  <img src="./docs/image_classifier_server.gif">
</p>


## Usage examples

### Using dedicated Dockerfile
```
FROM moreillon/image-classifier-server-fastapi
COPY path-to-your-model ./model
```
