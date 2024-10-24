# base image: NGC tf_image(Jetpack 5.1)
FROM nvcr.io/nvidia/l4t-tensorflow:r35.3.1-tf2.11-py3
# FROM python

WORKDIR /usr/src/app
COPY . .
RUN mkdir -p ./model
RUN mkdir -p ./preprocess

ENV http_proxy="http://10.115.1.14:5000/"
ENV https_proxy="http://10.115.1.14:5000/"
ENV HTTP_PROXY="http://10.115.1.14:5000/"
ENV HTTPS_PROXY="http://10.115.1.14:5000/"
ENV no_proxy="localhost,127.0.0.1"
ENV NO_PROXY="localhost,127.0.0.1"

# RUN apt update
# RUN apt install -y cmake libgl1 libglib2.0-0
# required to run opencv-python
#RUN apt-get update && apt-get -y install libgl1-mesa-dev libglib2.0-0

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN wget https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl
RUN pip3 install onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl

EXPOSE 80

CMD uvicorn main:app --host 0.0.0.0 --port 80