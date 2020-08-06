"""
Model class for YOLOv4
"""
import tensorflow as tf

from tf2_yolov4.anchors import compute_normalized_anchors
from tf2_yolov4.backbones.csp_darknet53 import csp_darknet53
from tf2_yolov4.heads.yolov3_head import yolov3_head
from tf2_yolov4.necks.yolov4_neck import yolov4_neck
from tf2_yolov4.tools.weights import get_weights_by_keyword_or_path


def YOLOv4(
    input_shape,
    num_classes,
    anchors,
    training=False,
    yolo_max_boxes=50,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.5,
    weights="darknet",
):
    """
    YOLOv4 Model

    Args:
        input_shape (Tuple[int]): Input shape of the image (H,W,C) . The Height and Width must be multiple of 32
        anchors (List[numpy.array[int, 2]]): List of 3 numpy arrays containing the anchor sizes used for each stage.
            The first and second columns of the numpy arrays contain respectively the width and the height of the
            anchors.
        num_classes (int): Number of classes.
        training (boolean): If False, will output boxes computed through YOLO regression and NMS, and YOLO features
            otherwise. Set it True for training, and False for inferences.
        yolo_max_boxes (int): Maximum number of boxes predicted on each image (across all anchors/stages)
        yolo_iou_threshold (float between 0. and 1.): IOU threshold defining whether close boxes will be merged
            during non max regression.
        yolo_score_threshold (float between 0. and 1.): Boxes with score lower than this threshold will be filtered
            out during non max regression.
        weights (str): one of `None` (random initialization),
            'darknet' (pre-training on COCO),
            or the path to the weights file to be loaded.
    Returns:
        tf.keras.Model: YoloV4 model

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If height and width in the input_shape  is not a multiple of 32

    """
    if (input_shape[0] % 32 != 0) | (input_shape[1] % 32 != 0):
        raise ValueError(
            f"Provided height and width in input_shape {input_shape} is not a multiple of 32"
        )

    backbone = csp_darknet53(input_shape)

    neck = yolov4_neck(input_shapes=backbone.output_shape)

    normalized_anchors = compute_normalized_anchors(anchors, input_shape)
    head = yolov3_head(
        input_shapes=neck.output_shape,
        anchors=normalized_anchors,
        num_classes=num_classes,
        training=training,
        yolo_max_boxes=yolo_max_boxes,
        yolo_iou_threshold=yolo_iou_threshold,
        yolo_score_threshold=yolo_score_threshold,
    )

    inputs = tf.keras.Input(shape=input_shape)
    lower_features = backbone(inputs)
    medium_features = neck(lower_features)
    upper_features = head(medium_features)

    yolov4 = tf.keras.Model(inputs=inputs, outputs=upper_features, name="YOLOv4")

    weights_path = get_weights_by_keyword_or_path(weights, model=yolov4)
    if weights_path is not None:
        yolov4.load_weights(str(weights_path), by_name=True, skip_mismatch=True)

    return yolov4
