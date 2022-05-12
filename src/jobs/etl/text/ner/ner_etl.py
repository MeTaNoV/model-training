import json
import argparse
import logging
from typing import Dict, Any

from labelbox import Client
from labelbox.data.annotation_types import TextEntity
from labelbox.data.annotation_types import Label

from training_lib.clients import get_lb_client, get_gcs_client
from training_lib.errors import InvalidLabelException
from training_lib.etl import get_labels_for_model_run, process_labels_in_threadpool, partition_mapping, validate_label, \
    validate_vertex_dataset
from training_lib.storage import upload_ndjson_data, create_gcs_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERTEX_MIN_TRAINING_EXAMPLES, VERTEX_MAX_TRAINING_EXAMPLES = 50, 100_000
MIN_ANNOTATIONS, MAX_ANNOTATIONS = 1, 20
MIN_ANNOTATION_NAME_LENGTH, MAX_ANNOTATION_NAME_LENGTH = 2, 30


def process_label(label: Label) -> Dict[str, Any]:
    """
    Function for converting a labelbox Label object into a vertex json label for ner.
    Args:
        label: the label to convert
    Returns:
        Dict representing a vertex label
    """
    text_annotations = []
    validate_label(label)
    for annotation in label.annotations:
        if isinstance(annotation.value, TextEntity):
            entity = annotation.value
            # TODO: Check this condition up front by looking at the ontology
            if len(annotation.name) < MIN_ANNOTATION_NAME_LENGTH or len(
                    annotation.name) > MAX_ANNOTATION_NAME_LENGTH:
                logger.warning(
                    "Skipping annotation `{annotation.name}`. "
                    "Length of name invalid. Must be: "
                    f"{MIN_ANNOTATION_NAME_LENGTH}<=num chars<={MAX_ANNOTATION_NAME_LENGTH}"
                )
                continue

            text_annotations.append({
                "startOffset": entity.start,
                "endOffset": entity.end + 1,
                "displayName": annotation.name
            })

    if not (MIN_ANNOTATIONS <= len(text_annotations) <= MAX_ANNOTATIONS):
        raise InvalidLabelException("Skipping label. Number of annotations is not in range: "
                    f"{MIN_ANNOTATIONS}<=num annotations<={MAX_ANNOTATIONS}")

    return {
        "textSegmentAnnotations": text_annotations,
        # Note that this always uploads the text data in-line
        "textContent": label.data.value,
        'dataItemResourceLabels': {
            "aiplatform.googleapis.com/ml_use": partition_mapping[label.extra.get("Data Split")],
            "dataRowId": label.data.uid
        }
    }


def ner_etl(lb_client: Client, model_run_id: str) -> str:
    """
    Creates a jsonl file that is used for input into a vertex ai training job

    This code barely validates the requirements as listed in the vertex documentation.
    Read more about the restrictions here:
        - https://cloud.google.com/vertex-ai/docs/datasets/prepare-text#entity-extraction

    Args:
        lb_client: Labelbox client object
        model_run_id: the id of the model run to export labels from
    Retuns:
        stringified ndjson
    """
    labels = get_labels_for_model_run(lb_client, model_run_id, media_type='text')
    vertex_labels = process_labels_in_threadpool(process_label, labels)
    validate_vertex_dataset(vertex_labels, 'textSegmentAnnotations', min_classes = 1, max_classes = 100, min_labels = 50)
    return "\n".join([json.dumps(label) for label in vertex_labels])


def main(model_run_id: str, gcs_bucket: str, gcs_key: str):
    lb_client = get_lb_client()
    bucket = get_gcs_client().bucket(gcs_bucket)
    json_data = ner_etl(lb_client, model_run_id)
    gcs_key = gcs_key or create_gcs_key('ner')
    etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
    logger.info("ETL Complete. URI: %s", f"{etl_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vertex AI ETL Runner')
    parser.add_argument('--gcs_bucket', type=str, required=True)
    parser.add_argument('--model_run_id', type=str, required=True)
    parser.add_argument('--gcs_key', type=str, required=False, default=None)
    args = parser.parse_args()
    main(**vars(args))
