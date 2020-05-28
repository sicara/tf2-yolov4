import pytest
import tensorflow as tf

from tf2_yolov4.model import YOLOv4


def test_model_should_predict_valid_shapes_at_training(yolov4_training, num_classes):
    n_images = 2

    bounding_box_shape = 4
    objectness_score_shape = 1
    expected_head_shape = (num_classes + objectness_score_shape) + bounding_box_shape

    output_1, output_2, output_3 = yolov4_training.predict(
        tf.random.uniform((n_images, 416, 416, 3))
    )

    assert output_1.shape == (n_images, 13, 13, 3, expected_head_shape)
    assert output_2.shape == (n_images, 26, 26, 3, expected_head_shape)
    assert output_3.shape == (n_images, 52, 52, 3, expected_head_shape)


def test_model_should_predict_valid_shapes_at_inference(
    yolov4_inference, yolo_max_boxes
):
    n_images = 2

    boxes, scores, classes, valid_detections = yolov4_inference.predict(
        tf.random.uniform((n_images, 416, 416, 3))
    )

    assert boxes.shape == (n_images, yolo_max_boxes, 4)
    assert scores.shape == (n_images, yolo_max_boxes)
    assert classes.shape == (n_images, yolo_max_boxes)
    assert valid_detections.shape == tuple([n_images])


def test_model_instanciation_should_fail_with_input_shapes_not_multiple_of_32():
    with pytest.raises(Exception):
        YOLOv4((32, 33, 3), 80, [])

    with pytest.raises(Exception):
        YOLOv4((33, 32, 3), 80, [])
