import argparse
import os
import numpy as np
import cv2
import mxnet as mx
from collections import namedtuple

import json
from google.cloud import storage
import pickle
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MXNET Resnet152 setup. Downloads weights file
# TODO: add weights file to the container
path = 'http://data.mxnet.io/models/imagenet-11k/'
_ = [mx.test_utils.download(path + 'resnet-152/resnet-152-symbol.json'),
 mx.test_utils.download(path + 'resnet-152/resnet-152-0000.params'),
 mx.test_utils.download(path + 'synset.txt')]

sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
mod.set_params(arg_params, aux_params)
all_layers = sym.get_internals()
fe_sym = all_layers['flatten0_output']

# Resnet model
model = mx.mod.Module(symbol=fe_sym, context=mx.cpu(), label_names=None)
model.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
model.set_params(arg_params, aux_params)

# KNN Model
x_train = []
y_train = []


def parse_uri(gs_uri):
    parts = gs_uri.replace("gs://", "").split("/")
    bucket_name, key = parts[0], "/".join(parts[1:])
    return bucket_name, key


def get_img_from_line(bucket, line):
    """
    Take an entry from an ETL jsonl file and return the image it refers to.
    Assume everything is in the same, given bucket. This is to save a bit
    of time and not call gcp_client.get_bucket() for each line.
    """
    _, key = parse_uri(line['imageGcsUri'])
    im_blob = bucket.get_blob(key)
    im_str = im_blob.download_as_string()

    # Read the image with opencv
    nparr = np.frombuffer(im_str, np.uint8)

    # return image, resized and transposed, ready for mxnet
    return np.transpose(cv2.resize(cv2.imdecode(nparr, cv2.IMREAD_COLOR),
                                   (224, 224)),
                        (2, 0, 1))


def compute_embeddings(images):
    """
    Compute embeddings for a bunch of images at a time
    Takes a list of opencv images (already transposed and resized)
    and returns a list of quantized embeddings
    """
    Batch = namedtuple('Batch', ['data'])
    model.forward(Batch([mx.nd.array(np.array(images))]))
    features = model.get_outputs()[0].asnumpy()
    features = np.squeeze(features)

    # These determine how the quantization happens
    FACTOR = 48
    CLAMP = 96
    OFFSET = 32

    embeddings = np.clip((features * FACTOR).astype(int), 0, CLAMP) + OFFSET

    return list(embeddings)


def main(gcs_bucket: str, etl_uri: str, model_file: str):
    gcs_client = storage.Client(project=os.environ['GOOGLE_PROJECT'])

    bucket = gcs_client.bucket(gcs_bucket)

    etl_bucket, etl_key = parse_uri(etl_uri)

    # We assume that etl_bucket==gcs_bucket, to save some time
    etl_blob = bucket.get_blob(etl_key)
    etl = [json.loads(l) for l in
           etl_blob.download_as_string().decode('UTF-8').split('\n')]

    # We only want the training lines for now
    etl_train = [line for line in etl
                 if line['dataItemResourceLabels']
                 ['aiplatform.googleapis.com/ml_use'] == 'train']

    # The resulting model will consist of three lists. Using lists makes
    # Training and evaluation with sklearn easier
    knn_model = {}
    knn_model['data_rows'] = [line['dataItemResourceLabels']['dataRowId']
                              for line in etl_train]
    knn_model['embeddings'] = []
    knn_model['classes'] = [line['classificationAnnotation']['displayName']
                              for line in etl_train]


    # We assume all these images are in the same bucket
    images = [get_img_from_line(bucket, line) for line in etl_train]

    # Now we compute embeddings, a few at a time because we can
    chunk_size = 100
    while images:
        knn_model['embeddings'] = knn_model['embeddings'] + \
                                  compute_embeddings(images[:chunk_size])
        images = images[chunk_size:]


    output_blob = bucket.blob(model_file)
    pickle_out = pickle.dumps(knn_model)
    output_blob.upload_from_string(pickle_out)

    logger.info("Training Complete. URI: %s", f"gs://{bucket.name}/{output_blob.name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN Classifier Custom Model Trainer')
    parser.add_argument('--gcs_bucket', type=str, required=True)
    parser.add_argument('--etl_uri', type=str, required=True)
    parser.add_argument('--model_file', type=str, required=False, default=None)
    args = parser.parse_args()
    main(**vars(args))
