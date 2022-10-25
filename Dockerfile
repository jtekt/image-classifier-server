FROM python:3.9

WORKDIR /usr/src/app

COPY . .

RUN apt-get update
RUN apt-get install -y cmake libgl1

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 80

CMD uvicorn main:app --host 0.0.0.0 --port 80
