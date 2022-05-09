from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Optional, Union, Literal
import logging

from training_lib.errors import InvalidDataRowException, InvalidLabelException

from labelbox.data.serialization import LBV1Converter
from labelbox.data.annotation_types import Label, LabelGenerator
import labelbox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping from labelbox partition names to vertex names
partition_mapping = {
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


def process_labels_in_threadpool(process_fn: Callable[..., str],labels: List[Label], *args, max_workers = 8) -> List[str]:
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
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        training_data_futures = (exc.submit(process_fn, label, *args) for label in labels)
        training_data = []
        filter_count = {'labels' : 0,'data_rows' : 0}
        for future in as_completed(training_data_futures):
            try:
                training_data.append(future.result())
            except InvalidDataRowException as e:
                logger.warning(f"Invalid data. %s" % str(e))
                filter_count['data_rows'] += 1
            except InvalidLabelException as e:
                logger.warning(f"Invalid label. %s" % str(e))
                filter_count['labels'] += 1
        logger.info("Filtered out %d data rows due to InvalidDataRowExceptions" % filter_count['data_rows'])
        logger.info("Filtered out %d data rows due to InvalidLabelException" % filter_count['labels'])
        return training_data


def get_labels_for_model_run(client: labelbox.Client, model_run_id: str, media_type: Optional[Union[Literal['image'], Literal['text']]] = None) -> List[LabelGenerator]:
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
    if media_type is not None:
        for row in json_labels:
            row['media_type'] = media_type
    return LBV1Converter.deserialize(json_labels)
