import json
import argparse
import logging
from typing import Literal, Union, Dict, Any
from collections import defaultdict

from training_lib.errors import InvalidLabelException, InvalidDatasetException
from training_lib.storage import upload_image_to_gcs, upload_ndjson_data,  \
    create_gcs_key, get_image_bytes
from training_lib.etl import process_labels_in_threadpool, get_labels_for_model_run, PARTITION_MAPPING, validate_label, \
    validate_vertex_dataset
from training_lib.clients import get_lb_client, get_gcs_client

from google.cloud import storage
from labelbox.data.annotation_types import Label, Checklist, Radio
from labelbox import Client


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_classification_label(label: Label,
                                        bucket: storage.Bucket, downsample_factor = 2.) -> Dict[str, Any]:
    """
    Function for converting a labelbox Label object into a vertex json label for text single classification.
    Args:
        label: the label to convert
        bucket: cloud storage bucket to write image data to
        downsample_factor: how much to scale the images by before running inference
    Returns:
        Dict representing a vertex label
    """
    classifications = []
    validate_label(label)
    image_bytes, _ = get_image_bytes(label.data.url, downsample_factor)

    for annotation in label.annotations:
        if isinstance(annotation.value, Radio):
            classifications.append({
                "displayName":
                    f"{annotation.name}_{annotation.value.answer.name}"
            })
    if len(classifications) > 1:
        raise InvalidLabelException(
            "Skipping example. Must provide <= 1 classification per image.")
    elif len(classifications) == 0:
        classification = {'displayName': 'no_label'}
    else:
        classification = classifications[0]

    gcs_uri = upload_image_to_gcs(image_bytes, label.data.uid, bucket)
    return {
        'imageGcsUri': gcs_uri,
        'classificationAnnotation': classification,
        'dataItemResourceLabels': {
            "aiplatform.googleapis.com/ml_use": PARTITION_MAPPING[label.extra.get("Data Split")],
            "dataRowId": label.data.uid
        }
    }


def process_multi_classification_label(label: Label,
                                       bucket: storage.Bucket, downsample_factor = 2.) -> Dict[str, Any]:
    """
        Function for converting a labelbox Label object into a vertex json label for text multi classification.
        Args:
            label: the label to convert
            bucket: cloud storage bucket to write image data to
            downsample_factor: how much to scale the images by before running inference
        Returns:
            Dict representing a vertex label
    """
    classifications = []
    validate_label(label)
    image_bytes, _ = get_image_bytes(label.data.url, downsample_factor)

    for annotation in label.annotations:
        if isinstance(annotation.value, Radio):
            classifications.append(
                f"{annotation.name}_{annotation.value.answer.name}")

        elif isinstance(annotation.value, Checklist):
            classifications.extend([{
                "displayName": f"{annotation.name}_{answer.name}"
            } for answer in annotation.value.answer])

    if len(classifications) == 0:
        classifications = [{'displayName': 'no_label'}]

    gcs_uri = upload_image_to_gcs(image_bytes, label.data.uid, bucket)

    return {
        'imageGcsUri': gcs_uri,
        'classificationAnnotations': classifications,
        'dataItemResourceLabels': {
            "aiplatform.googleapis.com/ml_use": PARTITION_MAPPING[label.extra.get("Data Split")],
            "dataRowId": label.data.uid
        }
    }

def image_classification_etl(lb_client: Client, model_run_id: str,
                             bucket: storage.Bucket, multi: bool) -> str:
    """
    Creates a jsonl file that is used for input into a vertex ai training job

    Read more about the configuration here::
        - Multi: https://cloud.google.com/vertex-ai/docs/datasets/prepare-image#multi-label-classification
        - Single: https://cloud.google.com/vertex-ai/docs/datasets/prepare-image#single-label-classification

    Args:
        lb_client: Labelbox client object
        model_run_id: the id of the model run to export labels from
        bucket: Cloud storage bucket used to upload image data to
        multi: boolean indicating whether or not the etl is for single or multi classification
    Retuns:
        stringified ndjson
    """

    labels = get_labels_for_model_run(lb_client, model_run_id, media_type='image')
    if multi:
        vertex_labels = process_labels_in_threadpool(process_multi_classification_label, labels, bucket)
        validate_vertex_dataset(vertex_labels, 'classificationAnnotations')
    else:
        vertex_labels = process_labels_in_threadpool(process_single_classification_label, labels, bucket)
        validate_vertex_dataset(vertex_labels, 'classificationAnnotation')
    return "\n".join([json.dumps(label) for label in vertex_labels])


def main(model_run_id: str, gcs_bucket: str, gcs_key: str,
         classification_type: Union[Literal['single'], Literal['multi']]):
    lb_client = get_lb_client()
    bucket = get_gcs_client().bucket(gcs_bucket)
    json_data = image_classification_etl(lb_client, model_run_id, bucket,
                                         classification_type == 'multi')
    gcs_key = gcs_key or create_gcs_key(f'image-{classification_type}-classification')
    etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
    logger.info("ETL Complete. URI: %s", f"{etl_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vertex AI ETL Runner')
    parser.add_argument('--gcs_bucket', type=str, required=True)
    parser.add_argument('--model_run_id', type=str, required=True)
    parser.add_argument('--classification_type',
                        choices=['single', 'multi'],
                        required=True)
    parser.add_argument('--gcs_key', type=str, required=False, default=None)
    args = parser.parse_args()
    main(**vars(args))
