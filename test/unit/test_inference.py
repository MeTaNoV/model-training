from unittest.mock import patch
import os
from contextlib import ExitStack

import pytest

from pipelines.images.bounding_box import BoundingBoxInference
from pipelines.images.classification import ImageClassificationInference
from pipelines.text.classification import TextClassificationInference
from pipelines.text.ner import NERInference

from types import SimpleNamespace

@pytest.fixture(scope="module", autouse=True)
def storage():
    with patch(
            "pipelines.types.storage",
            return_value = SimpleNamespace(
                Client = None
            )
    ):
        yield


def patch_inference_stage(inference_stage, batch_files):

    inference_stage.lb_client = SimpleNamespace(
        _get_single=lambda _obj, model_run_id: SimpleNamespace(
            model_id='fake-model-id',
            add_predictions=lambda name, annotations: SimpleNamespace(
                wait_until_done=lambda: None,
                errors=[]
            )
        )
    )

    with ExitStack() as stack:
        prediction_files = [stack.enter_context(open(batch_file, 'r')).read() for batch_file in batch_files]
        inference_stage.batch_predict = lambda *args, **kwargs: SimpleNamespace(
            iter_outputs=lambda: [
                SimpleNamespace(
                    download_as_string=lambda: prediction_file
                )
            for prediction_file in prediction_files]
        )
    inference_stage.export_model_run_labels = lambda model_run_id, data_type: []
    return inference_stage


def test_bbox_inference():
    os.environ['GOOGLE_PROJECT'] = 'N/A'
    inference_stage = BoundingBoxInference("fake-key")
    inference_stage = patch_inference_stage(inference_stage, ['test/unit/assets/bbox-test.txt'])
    inference_stage.get_tool_info = lambda model_id: {'Grapes': 'fake-feature-schema-id123'}
    inference_stage.run(None, None, None, None)

def test_text_single_classification_inference():
    os.environ['GOOGLE_PROJECT'] = 'N/A'


    inference_stage = TextClassificationInference("fake-key", 'single')
    inference_stage.get_answer_info = lambda model_id: {
        'positive or negative_positive': {
            'parent_feature_schema_id': '2' * 25,
            'feature_schema_id': '1' + '2' * 24,
            'type': 'radio'
        },
        'positive or negative_negative': {
            'parent_feature_schema_id': '2' * 25,
            'feature_schema_id': '0' + '2' * 24,
            'type' : 'radio'
        }
    }
    inference_stage = patch_inference_stage(inference_stage, ['test/unit/assets/text-single.txt'])
    inference_stage.run("gs://fake/etl/file.jsonl", None, None, None)


def test_text_multi_classification_inference():
    os.environ['GOOGLE_PROJECT'] = 'N/A'
    inference_stage = TextClassificationInference("fake-key", 'multi')
    inference_stage.get_answer_info = lambda model_id: {
        'course_NUM': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '1' + '0' * 24,
            'type' : 'checklist'
        },
        'course_HUM': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '2' + '0' * 24,
            'type': 'checklist'
        },
        "course_DESC": {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '3' + '0' * 24,
            'type': 'checklist'
        },
        "course_ENTY": {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '4' + '0' * 24,
            'type': 'checklist'
        },
        "course_LOC": {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '5' + '0' * 24,
            'type': 'checklist'
        },
        "fine_gr": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '0' + '1' * 24,
            'type': 'checklist'
        },
        "fine_country": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '1' + '1' * 24,
            'type': 'checklist'
        },
        "fine_desc": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '2' + '1' * 24,
            'type': 'checklist'
        },
        "fine_date": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '3' + '1' * 24,
            'type': 'checklist'
        },
        "fine_city": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '4' + '1' * 24,
            'type': 'checklist'
        },
        "fine_reason": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '5' + '1' * 24,
            'type': 'checklist'
        },
        "fine_def": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '6' + '1' * 24,
            'type': 'checklist'
        },
        "fine_ind": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '7' + '1' * 24,
            'type': 'checklist'
        },
        "fine_count": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '8' + '1' * 24,
            'type': 'checklist'
        },
        "fine_manner": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '9' + '1' * 24,
            'type': 'checklist'
        },
        "fine_cremat": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '10' + '1' * 23,
            'type': 'checklist'
        },
        "fine_other": {
            'parent_feature_schema_id': '1' * 25,
            'feature_schema_id': '110' + '1' * 22,
            'type': 'checklist'
        }
    }
    inference_stage = patch_inference_stage(inference_stage, ['test/unit/assets/text-multi.txt'])
    inference_stage.run("gs://fake/etl/file.jsonl", None, None, None)

def test_image_single_classification_inference():
    os.environ['GOOGLE_PROJECT'] = 'N/A'

    inference_stage = ImageClassificationInference("fake-key", 'single')
    inference_stage.get_answer_info = lambda model_id: {
        'Type of fruit_apples': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '0' + '2' * 24,
            'type': 'radio'
        },
        'Type of fruit_plums': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '1' + '2' * 24,
            'type': 'radio'
        },
        'Type of fruit_peaches': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '2' + '2' * 24,
            'type': 'radio'
        },
        'Type of fruit_strawberries': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '3' + '2' * 24,
            'type': 'radio'
        },
        'Type of fruit_oranges': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '4' + '2' * 24,
            'type': 'radio'
        },
        'Type of fruit_grapes': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '5' + '2' * 24,
            'type': 'radio'
        }
    }
    inference_stage = patch_inference_stage(inference_stage, ['test/unit/assets/image-single.txt'])
    inference_stage.run("gs://fake/etl/file.jsonl", None, None, None)

def test_image_multi_classification_inference():
    os.environ['GOOGLE_PROJECT'] = 'N/A'
    inference_stage = ImageClassificationInference("fake-key", 'multi')
    inference_stage.get_answer_info = lambda model_id: {
        'description_desert': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '0' + '2' * 24,
            'type': 'radio'
        },
        'description_mountains': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '1' + '2' * 24,
            'type': 'radio'
        },
        'description_sea': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '2' + '2' * 24,
            'type': 'radio'
        },
        'description_sunset': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '3' + '2' * 24,
            'type': 'radio'
        },
        'description_trees': {
            'parent_feature_schema_id': '0' * 25,
            'feature_schema_id': '4' + '2' * 24,
            'type': 'radio'
        }
    }
    inference_stage = patch_inference_stage(inference_stage, ['test/unit/assets/image-multi.txt'])
    inference_stage.run("gs://fake/etl/file.jsonl", None, None, None)



def test_ner_inference():
    os.environ['GOOGLE_PROJECT'] = 'N/A'
    inference_stage = NERInference("fake-key")
    inference_stage = patch_inference_stage(inference_stage, ['test/unit/assets/ner-1.txt']) #, 'test/unit/assets/ner-2.txt'])
    inference_stage.get_tool_info = lambda model_id: {
        'B-PER' : 'fake-feature-schema-id001',
        'B-ORG' : 'fake-feature-schema-id002',
        'B-MISC': 'fake-feature-schema-id003',
        'B-LOC': 'fake-feature-schema-id004',
        'I-PER' : 'fake-feature-schema-id005',
        'I-ORG': 'fake-feature-schema-id006',
        'I-LOC': 'fake-feature-schema-id007',
        'I-MISC': 'fake-feature-schema-id008',
    }
    inference_stage.run(None, None, None, None)