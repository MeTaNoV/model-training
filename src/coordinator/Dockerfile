FROM python:3.9-slim-buster
RUN apt-get update -y && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY coordinator/requirements.txt .
RUN pip3 install -r requirements.txt

COPY ./training_lib training_lib
RUN pip3 install training_lib/

ARG GCS_BUCKET
ENV GCS_BUCKET=$GCS_BUCKET

ARG GOOGLE_PROJECT
ENV GOOGLE_PROJECT=$GOOGLE_PROJECT

ARG GOOGLE_SERVICE_ACCOUNT
ENV GOOGLE_SERVICE_ACCOUNT=$GOOGLE_SERVICE_ACCOUNT

ARG DEPLOYMENT_NAME
ENV DEPLOYMENT_NAME=$DEPLOYMENT_NAME


COPY coordinator .

ENTRYPOINT [ "python3", "coordinator.py"]