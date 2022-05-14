from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Optional, Union, Literal, Dict, Any
import logging
from collections import Counter

from training_lib.errors import InvalidDataRowException, InvalidLabelException, InvalidDatasetException

from labelbox.data.serialization import LBV1Converter
from labelbox.data.annotation_types import Label
import labelbox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping from labelbox partition names to vertex names
PARTITION_MAPPING = {
    'training': 'train',
    'test': 'test',
    'validation': 'validation'
}


def validate_label(label: Label) -> None:
    """
    Checks if a label has the required fields for any etl pipeline.
    """
    if label.extra.get("Data Split") is None:
        raise InvalidLabelException(f"No data split assigned to label `{label.uid}`.")
    if not len(label.annotations):
        raise InvalidLabelException(f"No annotations found for label `{label.uid}`.")


def process_labels_in_threadpool(process_fn: Callable[..., Dict[str, Any]],labels: List[Label], *args, max_workers = 8) -> List[Dict[str, Any]]:
    """
    Function for running etl functions in parallel.
    Args:
        process_fn: Function to execute in parallel. Should accept Label as the first param and then any optional number of args.
        labels: List of labels to process
        args: Args that are passed through to the process_fn
        max_workers: How many threads should be used

    Returns:
        A list of results from the process_fn
    """
    vertex_labels = []
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        training_data_futures = (exc.submit(process_fn, label, *args) for label in labels)
        filter_count = {'labels' : 0,'data_rows' : 0}
        for future in as_completed(training_data_futures):
            try:
                vertex_labels.append(future.result())
            except InvalidDataRowException as e:
                logger.warning(f"Invalid data. %s" % str(e))
                filter_count['data_rows'] += 1
            except InvalidLabelException as e:
                logger.warning(f"Invalid label. %s" % str(e))
                filter_count['labels'] += 1

        logger.info("Filtered out %d data rows due to InvalidDataRowExceptions" % filter_count['data_rows'])
        logger.info("Filtered out %d data rows due to InvalidLabelException" % filter_count['labels'])
    return vertex_labels


def get_labels_for_model_run(client: labelbox.Client, model_run_id: str, media_type: Optional[Union[Literal['image'], Literal['text']]] = None):
    """
    Args:
        client: Labelbox client used for fetching labels
        model_run_id: model run to fetch labels for
        media_type: required when we can't tell what the media type is from the raw export.
            E.g. if the data row content doesn't have a file extension and the annotations are classifications
            we can't infer the type without fetching the contents.
    Returns:
        LabelGenerator with labels for the
    """
    model_run = client.get_model_run(model_run_id)
    json_labels = model_run.export_labels(download=True)
    for row in json_labels:
        if media_type is not None:
            row['media_type'] = media_type

        # Strip subclasses
        for annotation in row['Label']['objects']:
            if 'classifications' in annotation:
                del annotation['classifications']

    return LBV1Converter.deserialize(json_labels)


def _get_display_names_for_label(vertex_label, annotation_name):
    annotation_data = vertex_label[annotation_name]
    if isinstance(annotation_data, list):
        for annotation in annotation_data:
            yield annotation['displayName']
    else:
        yield annotation_data['displayName']

def validate_vertex_dataset(vertex_labels, annotation_name, min_labels_per_class = 10, min_classes = None, max_classes = None, min_labels = None):
    """
    Args:
        vertex_labels : A list of dictionaries in any vertex compatible format
        annotation_name: A string used for looking up the displayName for the annotation
            All vertex annotations have a top level key (annotation_name) that either contains a list of dicts or a dict with a displayName key
        min_labels_per_class: min number of labels that a class must exist in
        max_classes: Max total classes in the entire dataset
        min_labels: Minimum number of labels
    """

    class_counts = {partition: Counter() for partition in PARTITION_MAPPING.values()}
    for vertex_label in vertex_labels:
        partition = vertex_label['dataItemResourceLabels']['aiplatform.googleapis.com/ml_use']
        for display_name in _get_display_names_for_label(vertex_label, annotation_name):
            class_counts[partition][display_name] += 1

    for partition in class_counts:
        for class_name, example_count in class_counts[partition].items():
            if example_count < min_labels_per_class:
                raise InvalidDatasetException(
                    f"Not enough examples for `{class_name}` in the {partition} partition. Expected {min_labels_per_class}, Found {example_count}"
                )

    for partition in class_counts:
        if min_classes is not None and len(class_counts[partition]) < min_classes:
            raise InvalidDatasetException(f"Must provide at least {min_classes} classes. Partition {partition} only had {len(class_counts[partition])}")
        if max_classes is not None and len(class_counts[partition]) > max_classes:
            raise InvalidDatasetException(f"Must provide at no more than {max_classes} classes. Partition {partition} had {len(class_counts[partition])}")

    if min_labels is not None and  len(vertex_labels) < min_labels:
        raise InvalidLabelException(f"Must provide at least {min_labels} complete labels. Found {len(vertex_labels)}")
