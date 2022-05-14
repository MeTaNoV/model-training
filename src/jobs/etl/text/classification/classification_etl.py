import json
import argparse
import logging
from typing import Literal, Union, Dict, Any
from collections import Counter

from training_lib.errors import InvalidLabelException
from training_lib.etl import process_labels_in_threadpool, get_labels_for_model_run, PARTITION_MAPPING, validate_label, \
    validate_vertex_dataset
from training_lib.clients import get_lb_client, get_gcs_client
from training_lib.storage import upload_ndjson_data, create_gcs_key
from training_lib.storage import upload_text_to_gcs

from labelbox import Client
from labelbox.data.annotation_types import Label, Checklist, Radio
from google.cloud import storage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VERTEX_MIN_TRAINING_EXAMPLES = 50


def process_single_classification_label(label: Label,
                                        bucket: storage.Bucket) -> Dict[str, Any]:
    """
    Function for converting a labelbox Label object into a vertex json label for text single classification.
    Args:
        label: the label to convert
        bucket: cloud storage bucket to write text data to
    Returns:
        Dict representing a vertex label
    """
    classifications = []
    validate_label(label)
    for annotation in label.annotations:
        if isinstance(annotation.value, Radio):
            classifications.append({
                "displayName":
                    f"{annotation.name}_{annotation.value.answer.name}"
            })

    if len(classifications) > 1:
        raise InvalidLabelException(
            "Skipping example. Must provide <= 1 classification per text document."
        )

    elif len(classifications) == 0:
        classification = {'displayName': 'no_label'}
    else:
        classification = classifications[0]

    uri = upload_text_to_gcs(label.data.value, label.data.uid, bucket)
    return {
        'textGcsUri': uri,
        'classificationAnnotation': classification,
        'dataItemResourceLabels': {
            "aiplatform.googleapis.com/ml_use": PARTITION_MAPPING[label.extra.get("Data Split")],
            "dataRowId": label.data.uid
        }
    }


def process_multi_classification_label(label: Label,
                                       bucket: storage.Bucket) -> Dict[str, Any]:
    """
    Function for converting a labelbox Label object into a vertex json label for text multi classification.
    Args:
        label: the label to convert
        bucket: cloud storage bucket to write text data to
    Returns:
        Dict representing a vertex label
    """
    classifications = []
    validate_label(label)
    for annotation in label.annotations:
        if isinstance(annotation.value, Radio):
            classifications.append({
                "displayName":
                    f"{annotation.name}_{annotation.value.answer.name}"
            })
        elif isinstance(annotation.value, Checklist):
            # Display name is a combination of tool name and value.
            # This makes it so that tool names don't need to be globally unique
            classifications.extend([{
                "displayName": f"{annotation.name}_{answer.name}"
            } for answer in annotation.value.answer])

    if len(classifications) == 0:
        classifications = [{'displayName': 'no_label'}]

    uri = upload_text_to_gcs(label.data.value, label.data.uid, bucket)
    return {
        'textGcsUri': uri,
        'classificationAnnotations': classifications,
        'dataItemResourceLabels': {
            "aiplatform.googleapis.com/ml_use": PARTITION_MAPPING[label.extra.get("Data Split")],
            "dataRowId": label.data.uid
        }
    }


def text_classification_etl(lb_client: Client, model_run_id: str,
                            bucket: storage.Bucket, multi: bool) -> str:
    """
    Creates a jsonl file that is used for input into a vertex ai training job

    Read more about the configuration here:
        - Multi: https://cloud.google.com/vertex-ai/docs/datasets/prepare-text#multi-label-classification
        - Single: https://cloud.google.com/vertex-ai/docs/datasets/prepare-text#single-label-classification

    Args:
        lb_client: Labelbox client object
        model_run_id: the id of the model run to export labels from
        bucket: Cloud storage bucket used to upload text data to
        multi: boolean indicating whether or not the etl is for single or multi classification
    Retuns:
        stringified ndjson
    """

    labels = get_labels_for_model_run(lb_client, model_run_id, media_type='text')
    if multi:
        vertex_labels = process_labels_in_threadpool(process_multi_classification_label, labels, bucket)
        validate_vertex_dataset(vertex_labels, 'classificationAnnotations', min_classes=2, max_classes=5000)
    else:
        vertex_labels = process_labels_in_threadpool(process_single_classification_label, labels, bucket)
        validate_vertex_dataset(vertex_labels, 'classificationAnnotation', min_classes=2, max_classes=5000)
    return "\n".join([json.dumps(label) for label in vertex_labels])


def main(model_run_id: str, gcs_bucket: str, gcs_key: str,
         classification_type: Union[Literal['single'], Literal['multi']]):
    lb_client = get_lb_client()
    bucket = get_gcs_client().bucket(gcs_bucket)
    json_data = text_classification_etl(lb_client, model_run_id, bucket,
                                        classification_type == 'multi')
    gcs_key = gcs_key or create_gcs_key(f'text-{classification_type}-classification')
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
