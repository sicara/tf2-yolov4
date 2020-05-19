"""
Model class for YOLOv4
"""
import tensorflow as tf

from tf2_yolov4.backbones.csp_darknet53 import csp_darknet53
from tf2_yolov4.config.anchors import YOLOv4Config
from tf2_yolov4.heads.yolov3_head import yolov3_head
from tf2_yolov4.necks.yolov4_neck import yolov4_neck


def YOLOv4(
    input_shape,
    anchors,
    num_classes,
    training=False,
    yolo_max_boxes=50,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.8,
):
    """
    YOLOv4 Model

    Args:
        input_shape (Tuple[int]): Input shape of the image
        anchors (List[numpy.array[int, 2]]): List of 3 numpy arrays containing the anchor sizes used for each stage.
            The first and second columns of the numpy arrays contain respectively the height and the width of the
            anchors.
        num_classes (int): Number of classes.
        training (boolean): If False, will output boxes computed through YOLO regression and NMS, and YOLO features
            otherwise. Set it True for training, and False for inferences.
        yolo_max_boxes (int): Maximum number of boxes predicted on each image (across all anchors/stages)
        yolo_iou_threshold (float between 0. and 1.): IOU threshold defining whether close boxes will be merged
            during non max regression.
        yolo_score_threshold (float between 0. and 1.): Boxes with score lower than this threshold will be filtered
            out during non max regression.
    """
    backbone = csp_darknet53(input_shape)
    neck = yolov4_neck(input_shapes=backbone.output_shape)
    head = yolov3_head(
        input_shapes=neck.output_shape,
        anchors=anchors,
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

    return tf.keras.Model(inputs=inputs, outputs=upper_features, name="YOLOv4")


if __name__ == "__main__":
    model = YOLOv4(
        input_shape=(608, 416, 3),
        anchors=YOLOv4Config.get_yolov4_anchors(),
        num_classes=80,
    )

    outputs = model.predict(tf.random.uniform((16, 608, 416, 3)), steps=1)
    model.summary()
    for output in outputs:
        print(output.shape)
