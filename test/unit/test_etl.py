from unittest.mock import patch
import json

import numpy as np
import ndjson
import pytest

from images.classification.classification_etl import image_classification_etl
from text.classification.classification_etl import text_classification_etl
from images.bounding_box.bounding_box_etl import bounding_box_etl
from text.ner.ner_etl import ner_etl
from training_lib.errors import InvalidDatasetException
from training_lib.etl import PARTITION_MAPPING


def test_bounding_box_etl(bounding_box_label_generator, mock_bucket):
    with patch(
            "images.bounding_box.bounding_box_etl.get_labels_for_model_run",
            return_value=bounding_box_label_generator(10, 0,0,40,40),
    ), patch("images.bounding_box.bounding_box_etl.get_image_bytes", return_value = (np.zeros((50,50,3)), (50, 50))):
        result_ndjson = bounding_box_etl(None, "79547bc7-9fae-48bf-96b5-8a80c0b618e4", mock_bucket)
        result = ndjson.loads(result_ndjson)
        assert len(result) == 10
        assert json.dumps(result[0]) == json.dumps(
            {"imageGcsUri": "gs://mock_bucket/training/images/cl3ixhzs50003i9yjnzcbc32o_50_50.jpg",
             "boundingBoxAnnotations": [
                 {"displayName": "dog", "xMin": 0.0, "yMin": 0.0, "xMax": 0.8, "yMax": 0.8}]
                , "dataItemResourceLabels":
                 {"aiplatform.googleapis.com/ml_use": "train", "dataRowId": "cl3ixhzs50003i9yjnzcbc32o"}
             }
        )

def test_filter_boxes(bounding_box_label_generator, mock_bucket):
    # bboxes that are too small should be filtered out
    with patch(
            "images.bounding_box.bounding_box_etl.get_labels_for_model_run",
            return_value=bounding_box_label_generator(10, 0, 0, 1, 1) + bounding_box_label_generator(1, 0, 0, 50, 50),
    ), patch("images.bounding_box.bounding_box_etl.get_image_bytes", return_value=(np.zeros((50, 50, 3)), (50, 50))):
        with pytest.raises(InvalidDatasetException) as err:
            result_ndjson = bounding_box_etl(None, "79547bc7-9fae-48bf-96b5-8a80c0b618e4", mock_bucket)
        assert str(err.value) == 'Not enough examples for `dog` in the train partition. Expected 10, Found 1'


def test_image_single_classification_etl(image_radio_label_generator, mock_bucket):
    with patch(
                "images.classification.classification_etl.get_labels_for_model_run",
                return_value=image_radio_label_generator(10),
        ), patch("images.classification.classification_etl.get_image_bytes", return_value=(np.zeros((50, 50, 3)), (50, 50))):
        result_ndjson = image_classification_etl(None, "79547bc7-9fae-48bf-96b5-8a80c0b618e4",mock_bucket, multi = False)
        result = ndjson.loads(result_ndjson)
        assert len(result) == 10
        assert json.dumps(result[0]) == json.dumps(
            {'imageGcsUri': 'gs://mock_bucket/training/images/cl3ixhzs50003i9yjnzcbc32o.jpg',
             'classificationAnnotation': {'displayName': 'animal type_dog'},
             'dataItemResourceLabels': {'aiplatform.googleapis.com/ml_use':
                                            'train', 'dataRowId': 'cl3ixhzs50003i9yjnzcbc32o'}})



def test_image_multi_classification_etl(image_checklist_label_generator, mock_bucket):
    with patch(
                "images.classification.classification_etl.get_labels_for_model_run",
                return_value=image_checklist_label_generator(10, 'training'),
        ), patch("images.classification.classification_etl.get_image_bytes", return_value=(np.zeros((50, 50, 3)), (50, 50))):
        result_ndjson = image_classification_etl(None, "79547bc7-9fae-48bf-96b5-8a80c0b618e4",mock_bucket, multi = True)
        result = ndjson.loads(result_ndjson)
        assert len(result) == 10
        assert json.dumps(result[0]) == json.dumps(
            {'imageGcsUri': 'gs://mock_bucket/training/images/cl3ixhzs50003i9yjnzcbc32o.jpg',
             'classificationAnnotations': [{'displayName': 'attributes_4 legs'}, {'displayName': 'attributes_2 eyes'}],
             'dataItemResourceLabels': {'aiplatform.googleapis.com/ml_use': 'train',
                                        'dataRowId': 'cl3ixhzs50003i9yjnzcbc32o'}}
        )


def test_text_single_classification_etl(text_radio_label_generator, mock_bucket):
    with patch(
                "text.classification.classification_etl.get_labels_for_model_run",
                return_value=(text_radio_label_generator(10, 'training', 'dog') +
                              text_radio_label_generator(10, 'training', 'cat') +
                              text_radio_label_generator(10, 'test', 'dog') +
                              text_radio_label_generator(10, 'test', 'cat') +
                              text_radio_label_generator(10, 'validation', 'dog') +
                              text_radio_label_generator(10, 'validation', 'cat')
                )
        ):
        result_ndjson = text_classification_etl(None, "79547bc7-9fae-48bf-96b5-8a80c0b618e4",mock_bucket, multi = False)
        result = ndjson.loads(result_ndjson)
        assert len(result) == 60
        example = sorted(result, key=lambda d: (d['classificationAnnotation']['displayName'], d['dataItemResourceLabels']['aiplatform.googleapis.com/ml_use']))[0]
        assert json.dumps(example, sort_keys = True) == json.dumps(
            {'textGcsUri': 'gs://mock_bucket/training/text/cl3ixhzs50003i9yjnzcbc32a.txt',
             'classificationAnnotation': {'displayName': 'animal type_cat'},
             'dataItemResourceLabels': {'aiplatform.googleapis.com/ml_use': 'test',
                                        'dataRowId': 'cl3ixhzs50003i9yjnzcbc32a'}},
        sort_keys = True)



def test_text_multi_classification_etl(text_checklist_label_generator, mock_bucket):
    with patch(
                "text.classification.classification_etl.get_labels_for_model_run",
                return_value=text_checklist_label_generator(10, "training") + text_checklist_label_generator(10, "test") + text_checklist_label_generator(10, "validation"),
        ):
        result_ndjson = text_classification_etl(None, "79547bc7-9fae-48bf-96b5-8a80c0b618e4",mock_bucket, multi = True)
        result = ndjson.loads(result_ndjson)
        assert len(result) == 30
        example = sorted(result, key=lambda d: d['dataItemResourceLabels']['aiplatform.googleapis.com/ml_use'])[0]
        assert json.dumps(example, sort_keys=True) == json.dumps(
            {'textGcsUri': 'gs://mock_bucket/training/text/cl3ixhzs50003i9yjnzcbc32a.txt', 'classificationAnnotations': [{'displayName': 'attributes_4 legs'}, {'displayName': 'attributes_2 eyes'}], 'dataItemResourceLabels': {'aiplatform.googleapis.com/ml_use': 'test', 'dataRowId': 'cl3ixhzs50003i9yjnzcbc32a'}}
        , sort_keys=True)



def test_ner_etl(ner_label_generator):
    with patch(
                "text.ner.ner_etl.get_labels_for_model_run",
                return_value=( ner_label_generator(40, "training") + ner_label_generator(10, "test") + ner_label_generator(10, "validation"))
        ):
        result_ndjson = ner_etl(None, "79547bc7-9fae-48bf-96b5-8a80c0b618e4")
        result = ndjson.loads(result_ndjson)
        assert len(result) == 60
        example = sorted(result, key=lambda d: d['dataItemResourceLabels']['aiplatform.googleapis.com/ml_use'])[0]
        assert json.dumps(example, sort_keys=True) == json.dumps(
            {'textSegmentAnnotations': [{'startOffset': 0, 'endOffset': 6, 'displayName': 'person'}],
             'textContent': 'some string data', 'dataItemResourceLabels': {'aiplatform.googleapis.com/ml_use': 'test',
                                                                           'dataRowId': 'cl3ixhzs50003i9yjnzcbc32a'}}
        , sort_keys=True)


def test_partition_min_examples(bounding_box_label_generator, mock_bucket):
    # Each partition should have at least n items..
    for data_split in ['training', 'test', 'validation']:
        with patch(
                "images.bounding_box.bounding_box_etl.get_labels_for_model_run",
                return_value=bounding_box_label_generator(1, 0, 0, 50, 50, data_split = data_split),
        ), patch("images.bounding_box.bounding_box_etl.get_image_bytes", return_value=(np.zeros((50, 50, 3)), (50, 50))):
            with pytest.raises(InvalidDatasetException) as err:
                bounding_box_etl(None, "79547bc7-9fae-48bf-96b5-8a80c0b618e4", mock_bucket)
            assert str(err.value).startswith(f'Not enough examples for `dog` in the {PARTITION_MAPPING[data_split]} partition. Expected ')

def test_min_number_of_labels(ner_label_generator):
    with patch(
                "text.ner.ner_etl.get_labels_for_model_run",
                return_value=( ner_label_generator(10, "training") + ner_label_generator(10, "test") + ner_label_generator(10, "validation"))
        ):
        with pytest.raises(InvalidDatasetException) as err:
            ner_etl(None, "79547bc7-9fae-48bf-96b5-8a80c0b618e4")
        assert str(err.value) == "Must provide at least 50 complete labels. Found 30"


def test_min_class_counts(text_radio_label_generator, mock_bucket):
    with patch(
                "text.classification.classification_etl.get_labels_for_model_run",
                return_value=(text_radio_label_generator(10, 'training', 'dog') +
                              text_radio_label_generator(10, 'test', 'dog') +
                              text_radio_label_generator(10, 'validation', 'dog'))

        ):
        with pytest.raises(InvalidDatasetException) as err:
            text_classification_etl(None, "79547bc7-9fae-48bf-96b5-8a80c0b618e4",mock_bucket, False)
        assert str(err.value) == "Must provide at least 2 classes. Partition train only had 1"