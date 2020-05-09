import pytest

from tf2_yolov4.backbones.backbone import CSPDarknet53


@pytest.fixture(scope="session")
def cspdarknet53_416():
    return CSPDarknet53((416, 416, 3))
