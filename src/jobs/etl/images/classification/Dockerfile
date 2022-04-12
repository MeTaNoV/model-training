FROM python:3.9-slim-buster

RUN apt-get update -y && apt-get install -y libgl1-mesa-glx
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG DEPLOYMENT_NAME
ENV DEPLOYMENT_NAME=$DEPLOYMENT_NAME

COPY . .

ENTRYPOINT [ "python3", "classification_etl.py"]
