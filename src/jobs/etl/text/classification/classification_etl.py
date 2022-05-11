import json
import argparse
import logging
from typing import Literal, Union
from collections import Counter

from training_lib.errors import InvalidLabelException
from training_lib.etl import process_labels_in_threadpool, get_labels_for_model_run, partition_mapping, validate_label
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
                                        bucket: storage.Bucket) -> str:
    """
    Function for converting a labelbox Label object into a vertex json label for text single classification.
    Args:
        label: the label to convert
        bucket: cloud storage bucket to write text data to
    Returns:
        Stringified json representing a vertex label
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
    return json.dumps({
        'textGcsUri': uri,
        'classificationAnnotation': classification,
        'dataItemResourceLabels': {
            "aiplatform.googleapis.com/ml_use": partition_mapping[label.extra.get("Data Split")],
            "dataRowId": label.data.uid
        }
    })


def process_multi_classification_label(label: Label,
                                       bucket: storage.Bucket) -> str:
    """
    Function for converting a labelbox Label object into a vertex json label for text multi classification.
    Args:
        label: the label to convert
        bucket: cloud storage bucket to write text data to
    Returns:
        Stringified json representing a vertex label
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
    return json.dumps({
        'textGcsUri': uri,
        'classificationAnnotations': classifications,
        'dataItemResourceLabels': {
            "aiplatform.googleapis.com/ml_use": partition_mapping[label.extra.get("Data Split")],
            "dataRowId": label.data.uid
        }
    })


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
        training_data = process_labels_in_threadpool(process_multi_classification_label, labels, bucket)
    else:
        training_data = process_labels_in_threadpool(process_single_classification_label, labels, bucket)

    if len(training_data) < VERTEX_MIN_TRAINING_EXAMPLES:
        raise InvalidLabelException("Not enough training examples provided.")

    if multi:
        training_data = list(filter_classes_by_example_count(training_data))
    return "\n".join(training_data)


def filter_classes_by_example_count(training_data, min_examples=30):
    names = []
    for row in training_data:
        for name in [
                x['displayName']
                for x in json.loads(row)['classificationAnnotations']
        ]:
            names.append(name)

    not_useable = {k for k, v in Counter(names).items() if v < min_examples}
    for row in training_data:
        names = [
            x['displayName']
            for x in json.loads(row)['classificationAnnotations']
        ]
        if not_useable.intersection(set(names)):
            continue
        else:
            yield row


def main(model_run_id: str, gcs_bucket: str, gcs_key: str,
         classification_type: Union[Literal['single'], Literal['multi']]):
    lb_client = get_lb_client()
    bucket = get_gcs_client().bucket(gcs_bucket)
    json_data = text_classification_etl(lb_client, model_run_id, bucket,
                                        classification_type == 'multi')
    gcs_key = gcs_key or create_gcs_key(f'{classification_type}-classification')
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
