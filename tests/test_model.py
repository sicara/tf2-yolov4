from pathlib import Path

import pytest
import tensorflow as tf

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
from tf2_yolov4.tools import weights


def test_model_should_predict_valid_shapes_at_training(yolov4_training, num_classes):
    n_images = 2

    bounding_box_shape = 4
    objectness_score_shape = 1
    expected_head_shape = (num_classes + objectness_score_shape) + bounding_box_shape

    output_1, output_2, output_3 = yolov4_training.predict(
        tf.random.uniform((n_images, 416, 416, 3))
    )

    assert output_1.shape == (n_images, 52, 52, 3, expected_head_shape)
    assert output_2.shape == (n_images, 26, 26, 3, expected_head_shape)
    assert output_3.shape == (n_images, 13, 13, 3, expected_head_shape)


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


@pytest.mark.parametrize("input_shape", [(32, 33, 3), (33, 32, 3)])
def test_model_instanciation_should_fail_with_input_shapes_not_multiple_of_32(
    input_shape,
):
    with pytest.raises(ValueError):
        YOLOv4(input_shape, 80, [])


def test_should_raise_error_if_weights_argument_is_unknown():
    with pytest.raises(ValueError):
        YOLOv4(input_shape=(416, 416, 3), num_classes=80, anchors=[], weights="unknown")


def test_should_download_pretrained_weight_if_not_available(mocker):
    mocker.patch("tf2_yolov4.model.csp_darknet53")
    mocker.patch("tf2_yolov4.model.compute_normalized_anchors")
    mocker.patch("tf2_yolov4.model.yolov4_neck")
    mocker.patch("tf2_yolov4.model.yolov3_head")
    mocker.patch(
        "tf2_yolov4.model.tf.keras.Model"
    ).return_value = mocker.sentinel.yolov4
    mocker.sentinel.yolov4.load_weights = mocker.MagicMock()

    mock_download_darknet_weights = mocker.patch(
        "tf2_yolov4.tools.weights.download_darknet_weights"
    )

    YOLOv4(
        input_shape=(416, 416, 3),
        num_classes=80,
        anchors=YOLOV4_ANCHORS,
        weights="darknet",
    )
    mock_download_darknet_weights.assert_called_once_with(mocker.sentinel.yolov4)


def test_should_load_pretrained_weights_if_available(mocker):
    mocker.patch("tf2_yolov4.model.csp_darknet53")
    mocker.patch("tf2_yolov4.model.compute_normalized_anchors")
    mocker.patch("tf2_yolov4.model.yolov4_neck")
    mocker.patch("tf2_yolov4.model.yolov3_head")
    mocker.patch(
        "tf2_yolov4.model.tf.keras.Model"
    ).return_value = mocker.sentinel.yolov4
    mocker.sentinel.yolov4.load_weights = mocker.MagicMock()

    mocker.patch.object(
        weights, "DARKNET_WEIGHTS_PATH", mocker.MagicMock(is_file=lambda: True)
    )
    mock_download_darknet_weights = mocker.patch(
        "tf2_yolov4.tools.weights.download_darknet_weights"
    )

    YOLOv4(
        input_shape=(416, 416, 3),
        num_classes=80,
        anchors=YOLOV4_ANCHORS,
        weights="darknet",
    )
    assert mock_download_darknet_weights.call_count == 0


def test_should_load_weights_from_file_if_path_exists(mocker):
    mocker.patch("tf2_yolov4.model.csp_darknet53")
    mocker.patch("tf2_yolov4.model.compute_normalized_anchors")
    mocker.patch("tf2_yolov4.model.yolov4_neck")
    mocker.patch("tf2_yolov4.model.yolov3_head")
    mocker.patch(
        "tf2_yolov4.model.tf.keras.Model"
    ).return_value = mocker.sentinel.yolov4
    mocker.sentinel.yolov4.load_weights = mocker.MagicMock()

    YOLOv4(
        input_shape=(416, 416, 3),
        num_classes=80,
        anchors=YOLOV4_ANCHORS,
        weights=Path(__file__),
    )
    mocker.sentinel.yolov4.load_weights.assert_called_once_with(
        str(Path(__file__)), by_name=True, skip_mismatch=True,
    )
