"""
Model class for YOLOv4
"""
import tensorflow as tf

from tf2_yolov4.backbones.csp_darknet53 import csp_darknet53
from tf2_yolov4.config.anchors import YOLOv4Config
from tf2_yolov4.heads.yolov3_head import yolov3_head
from tf2_yolov4.necks.yolov4_neck import yolov4_neck


class YOLOv4(tf.keras.Model):
    """YOLOv4 Model"""

    def __init__(self, input_shape, anchors, num_classes, predict_boxes=False):
        """
        Constructor

        Args:
            input_shape (Tuple[int]): Input shape of the image
            anchors (List[numpy.array[int, 2]]): List of 3 numpy arrays containing the anchor sizes used for each stage.
                The first and second columns of the numpy arrays contain respectively the height and the width of the
                anchors.
            num_classes (int): Number of classes.
        """
        super(YOLOv4, self).__init__(name="YOLOv4")
        self.backbone = csp_darknet53(input_shape)
        self.neck = yolov4_neck(input_shapes=self.backbone.output_shape)
        self.head = yolov3_head(
            input_shapes=self.neck.output_shape,
            anchors=anchors,
            num_classes=num_classes,
            predict_boxes=predict_boxes,
        )

    def call(self, inputs, training=False, mask=None):
        """
        YOLOv4's forward pass

        Args:
            inputs (tf.Tensor): 4D (N,H,W,C) input tensor
        """
        lower_features = self.backbone(inputs)
        medium_features = self.neck(lower_features)
        upper_features = self.head(medium_features)

        return upper_features


if __name__ == "__main__":

    model = YOLOv4(
        input_shape=(416, 832, 3),
        anchors=YOLOv4Config.get_yolov4_anchors(),
        num_classes=80,
        predict_boxes=True,
    )

    outputs = model.predict(tf.random.uniform((16, 416, 832, 3)))
    model.summary()
    for output in outputs:
        print(output.shape)
