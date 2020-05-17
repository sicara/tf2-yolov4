"""
Model class for YOLOv4
"""
import tensorflow as tf

from tf2_yolov4.backbones.csp_darknet53 import csp_darknet53
from tf2_yolov4.config.anchors import YOLOv4Config
from tf2_yolov4.heads.yolov3_head import yolov3_head
from tf2_yolov4.necks.yolov4_neck import yolov4_neck


def YOLOv4(input_shape, anchors, num_classes):
    """
    YOLOv4 Model

    Args:
        input_shape (Tuple[int]): Input shape of the image
        anchors (List[numpy.array[int, 2]]): List of 3 numpy arrays containing the anchor sizes used for each stage.
            The first and second columns of the numpy arrays contain respectively the height and the width of the
            anchors.
        num_classes (int): Number of classes.
    """
    backbone = csp_darknet53(input_shape)
    neck = yolov4_neck(input_shapes=backbone.output_shape)
    head = yolov3_head(
        input_shapes=neck.output_shape, anchors=anchors, num_classes=num_classes,
    )

    inputs = tf.keras.Input(shape=input_shape)
    lower_features = backbone(inputs)
    medium_features = neck(lower_features)
    upper_features = head(medium_features)

    return tf.keras.Model(inputs=inputs, outputs=upper_features, name="YOLOv4")


if __name__ == "__main__":
    model = YOLOv4(
        input_shape=(608, 608, 3),
        anchors=YOLOv4Config.get_yolov4_anchors(),
        num_classes=80,
    )

    outputs = model.predict(tf.random.uniform((16, 608, 608, 3)), steps=1)
    model.summary()
    for output in outputs:
        print(output.shape)
