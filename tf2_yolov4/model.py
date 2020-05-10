"""
Model class for YOLOv4
"""
import numpy as np
import tensorflow as tf

from tf2_yolov4.backbones.csp_darknet53 import CSPDarknet53
from tf2_yolov4.heads.yolov3_head import YOLOv3_head
from tf2_yolov4.necks.yolov4_neck import YOLOv4_neck


class YOLOv4(tf.keras.Model):
    """YOLOv4 Model"""

    def __init__(self, input_shape, anchors, num_classes):
        """
        Constructor

        Args:
            input_shape (Tuple[int]): Input shape of the image
        """
        super(YOLOv4, self).__init__(name="YOLOv4")
        self.backbone = CSPDarknet53(input_shape)
        self.neck = YOLOv4_neck(input_shapes=self.backbone.output_shape)
        self.head = YOLOv3_head(input_shapes=self.neck.output_shape, anchors=anchors, num_classes=num_classes)

    def call(self, inputs, training=None, mask=None):
        lower_features = self.backbone(inputs)
        medium_features = self.neck(lower_features)
        upper_features = self.head(medium_features)
        return upper_features


if __name__ == "__main__":
    yolov4_anchors = [
        np.array([(142, 110), (192, 243), (459, 401)], np.float32) / 416,
        np.array([(36, 75), (76, 55), (72, 146)], np.float32) / 416,
        np.array([(12, 16), (19, 36), (40, 28)], np.float32) / 416
    ]

    model = YOLOv4(input_shape=(416, 416, 3), anchors=yolov4_anchors, num_classes=80)

    outputs = model.predict(tf.random.uniform((16, 416, 416, 3)))
    model.summary()
    [print(output.shape) for output in outputs]
