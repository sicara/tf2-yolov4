"""
Contains original YOLO anchors and associated utils
"""
import numpy as np

YOLOV4_ANCHORS = [
    np.array([(12, 16), (19, 36), (40, 28)], np.float32),
    np.array([(36, 75), (76, 55), (72, 146)], np.float32),
    np.array([(142, 110), (192, 243), (459, 401)], np.float32),
]

YOLOV3_ANCHORS = [
    np.array([(10, 13), (16, 30), (33, 23)], np.float32),
    np.array([(30, 61), (62, 45), (59, 119)], np.float32),
    np.array([(116, 90), (156, 198), (373, 326)], np.float32),
]


def compute_normalized_anchors(anchors, input_shape):
    """
    Compute anchors resizing based on the architecture input shapes
    Args:
        anchors (List[numpy.array[int, 2]]): List of 3 numpy arrays containing the anchor sizes used for
            each stage. The first and second columns of the numpy arrays respectively contain the anchors width and
            height.
        input_shape (Tuple[int]): Input shape of the Network

    Returns:
        (List[numpy.array[int, 2]]): anchors resized based on the input shape of the Network.
    """
    height, width = input_shape[:2]
    return [anchor / np.array([width, height]) for anchor in anchors]
