import numpy as np
import pytest

from tf2_yolov4.backbones.csp_darknet53 import csp_darknet53
from tf2_yolov4.heads.yolov3_head import yolov3_head
from tf2_yolov4.necks.yolov4_neck import yolov4_neck


@pytest.fixture(scope="session")
def cspdarknet53_416():
    return csp_darknet53((416, 416, 3))


@pytest.fixture(scope="session")
def yolov4_neck_416():
    input_shapes = [
        (13, 13, 1024),
        (26, 26, 512),
        (52, 52, 256)
    ]
    return yolov4_neck(input_shapes)


@pytest.fixture(scope="session")
def yolov3_head_416():
    input_shapes = [
        (13, 13, 512),
        (26, 26, 256),
        (52, 52, 128)
    ]

    anchors = [
        np.array([(142, 110), (192, 243), (459, 401)], np.float32) / 416,
        np.array([(36, 75), (76, 55), (72, 146)], np.float32) / 416,
        np.array([(12, 16), (19, 36), (40, 28)], np.float32) / 416
    ]
    return yolov3_head(input_shapes, anchors=anchors, num_classes=80)
