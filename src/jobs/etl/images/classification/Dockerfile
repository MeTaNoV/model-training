FROM python:3.9-slim-buster

RUN apt-get update -y && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app
COPY ./jobs/etl/images/classification/requirements.txt .
RUN pip install -r requirements.txt

ARG DEPLOYMENT_NAME
ENV DEPLOYMENT_NAME=$DEPLOYMENT_NAME

COPY ./training_lib training_lib
RUN pip3 install training_lib/


COPY ./jobs/etl/images/classification/ .

ENTRYPOINT [ "python3", "classification_etl.py"]
