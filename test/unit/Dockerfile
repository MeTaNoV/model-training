FROM python:3.9-slim-buster

RUN apt-get update -y && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app
RUN pip install google-cloud-storage \
  google-cloud-secret-manager \
  opencv-python-headless==4.5.5.62 \
  labelbox[data] \
  pytest \
  google-cloud-aiplatform

COPY . .
