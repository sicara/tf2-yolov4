import pytest

from tf2_yolov4.backbones.csp_darknet53 import csp_darknet53


@pytest.fixture(scope="session")
def cspdarknet53_416():
    return csp_darknet53((416, 416, 3))
