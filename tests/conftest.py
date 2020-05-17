import pytest

from tf2_yolov4.backbones.csp_darknet53 import csp_darknet53
from tf2_yolov4.config.anchors import YOLOv4Config
from tf2_yolov4.heads.yolov3_head import yolov3_head
from tf2_yolov4.model import YOLOv4
from tf2_yolov4.necks.yolov4_neck import yolov4_neck


@pytest.fixture(scope="session")
def cspdarknet53_416():
    return csp_darknet53((416, 416, 3))


@pytest.fixture(scope="session")
def yolov4_neck_416():
    input_shapes = [(13, 13, 1024), (26, 26, 512), (52, 52, 256)]
    return yolov4_neck(input_shapes)


@pytest.fixture(scope="session")
def yolov3_head_416_training(num_classes, yolo_max_boxes):
    input_shapes = [(13, 13, 512), (26, 26, 256), (52, 52, 128)]

    anchors = YOLOv4Config.get_yolov4_anchors()
    return yolov3_head(
        input_shapes,
        anchors=anchors,
        num_classes=num_classes,
        predict_boxes=False,
        yolo_max_boxes=yolo_max_boxes,
        yolo_iou_threshold=0.5,
        yolo_score_threshold=0.8,
    )


@pytest.fixture(scope="session")
def yolov3_head_416_inference(num_classes, yolo_max_boxes):
    input_shapes = [(13, 13, 512), (26, 26, 256), (52, 52, 128)]

    anchors = YOLOv4Config.get_yolov4_anchors()
    return yolov3_head(
        input_shapes,
        anchors=anchors,
        num_classes=num_classes,
        predict_boxes=True,
        yolo_max_boxes=yolo_max_boxes,
        yolo_iou_threshold=0.5,
        yolo_score_threshold=0.8,
    )


@pytest.fixture(scope="session")
def num_classes():
    return 40


@pytest.fixture(scope="session")
def yolo_max_boxes():
    return 20


@pytest.fixture(scope="session")
def yolov4_training(
    cspdarknet53_416, yolov4_neck_416, yolov3_head_416_training, session_mocker
):
    session_mocker.patch(
        "tf2_yolov4.model.csp_darknet53"
    ).return_value = cspdarknet53_416
    session_mocker.patch("tf2_yolov4.model.yolov4_neck").return_value = yolov4_neck_416
    session_mocker.patch(
        "tf2_yolov4.model.yolov3_head"
    ).return_value = yolov3_head_416_training

    return YOLOv4(input_shape=(416, 416, 3), anchors=None, num_classes=0)


@pytest.fixture(scope="session")
def yolov4_inference(
    cspdarknet53_416, yolov4_neck_416, yolov3_head_416_inference, session_mocker
):
    session_mocker.patch(
        "tf2_yolov4.model.csp_darknet53"
    ).return_value = cspdarknet53_416
    session_mocker.patch("tf2_yolov4.model.yolov4_neck").return_value = yolov4_neck_416
    session_mocker.patch(
        "tf2_yolov4.model.yolov3_head"
    ).return_value = yolov3_head_416_inference

    return YOLOv4(input_shape=(416, 416, 3), anchors=None, num_classes=0)
