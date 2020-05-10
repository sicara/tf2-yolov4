import pytest

from tf2_yolov4.backbones.csp_darknet53 import CSPDarknet53


@pytest.fixture(scope="session")
def cspdarknet53_416():
    return CSPDarknet53((416, 416, 3))
