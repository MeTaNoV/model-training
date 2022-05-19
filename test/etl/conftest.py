from types import SimpleNamespace

import pytest

from labelbox.data.annotation_types import *


@pytest.fixture
def mock_bucket():
    return SimpleNamespace(
        name = "mock_bucket",
        blob = lambda key: SimpleNamespace(
            name = key,
            upload_from_file = lambda content, *args, **kwargs:  None,
            upload_from_string = lambda *args, **kwargs: None
        )
    )

@pytest.fixture
def image_data():
    return ImageData(uid = 'cl3ixhzs50003i9yjnzcbc32o')

@pytest.fixture
def text_data():
    return TextData(text = "some string data", uid='cl3ixhzs50003i9yjnzcbc32a')

@pytest.fixture
def label_builder():
    def get_label(data, annotations, data_split):
        return Label(data=data, annotations=
            annotations
        , extra={"Data Split": data_split})
    return get_label


@pytest.fixture
def radio_label_builder(label_builder):
    def get_radio_label(data, data_split, answer = "dog"):
        return label_builder(data,
            [
                        ClassificationAnnotation(
                            name = "animal type",
                            value = Radio(
                    answer = ClassificationAnswer(name = answer)
                             )),
            ], data_split)
    return get_radio_label

@pytest.fixture
def checklist_label_builder(label_builder):
    def get_checklist_label(data, data_split):
        return label_builder(data,
            [
                ClassificationAnnotation(name="attributes",
                                         value=Checklist(
                                             answer=[ClassificationAnswer(name='4 legs'),
                                                     ClassificationAnswer(name='2 eyes')]
                                         ))], data_split)
    return get_checklist_label


@pytest.fixture
def bounding_box_label_generator(label_builder, image_data):
    def get_bboxes(num_annotations, x0, y0, x1, y1, data_split = "training"):
        return [label_builder(data = image_data, annotations = [
            ObjectAnnotation(name = "dog",
                             value = Rectangle(
                                 start = Point(x = x0,y = y0),
                                 end = Point(x = x1,y = y1))
                             )
        ], data_split = data_split) for _ in range(num_annotations)]
    return get_bboxes

@pytest.fixture
def image_radio_label_generator(radio_label_builder, image_data):
    def get_radios(num_annotations, data_split = "training"):
        return [radio_label_builder(image_data, data_split) for _ in range(num_annotations)]
    return get_radios

@pytest.fixture
def image_checklist_label_generator(checklist_label_builder, image_data):
    def get_checklists(num_annotations, data_split = "training"):
        return [checklist_label_builder(image_data, data_split) for _ in range(num_annotations)]
    return get_checklists


@pytest.fixture
def text_radio_label_generator(radio_label_builder, text_data):
    def get_radios(num_annotations, data_split = "training", answer = "dog"):
        return [radio_label_builder(text_data, data_split, answer) for _ in range(num_annotations)]
    return get_radios

@pytest.fixture
def text_checklist_label_generator(checklist_label_builder, text_data):
    def get_checklists(num_annotations, data_split = "training"):
        return [checklist_label_builder(text_data, data_split) for _ in range(num_annotations)]
    return get_checklists


@pytest.fixture
def ner_label_generator(label_builder, text_data):
    def get_entities(num_annotations, data_split = "training"):
        return [label_builder(data = text_data, annotations = [
            ObjectAnnotation(name = "person",
                             value = TextEntity(
                                 start = 0,
                                 end = 5
                             ))
        ], data_split = data_split) for _ in range(num_annotations)]
    return get_entities
