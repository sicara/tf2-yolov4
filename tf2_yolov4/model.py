"""
Model class for YOLOv4
"""
import tensorflow as tf

from tf2_yolov4.backbones.CSPDarknet53 import CSPDarknet53
from tf2_yolov4.necks.yolov4_neck import YOLOv4_neck


class YOLOv4(tf.keras.Model):
    """YOLOv4 Model"""

    def __init__(self, input_shape):
        """
        Constructor

        Args:
            input_shape (Tuple[int]): Input shape of the image
        """
        super(YOLOv4, self).__init__()
        self.backbone = CSPDarknet53(input_shape)
        self.neck = YOLOv4_neck(input_shapes=self.backbone.output_shape)

    def call(self, inputs, training=None, mask=None):
        lower_features = self.backbone(inputs)
        upper_features = self.neck(lower_features)
        return upper_features


if __name__ == "__main__":
    model = YOLOv4((416, 416, 3))
    outputs = model.predict(tf.random.uniform((16, 416, 416, 3)))
    [print(output.shape) for output in outputs]
