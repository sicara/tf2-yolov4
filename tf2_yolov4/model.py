"""
Model class for YOLOv4
"""
import tensorflow as tf

from tf2_yolov4.backbone import CSPDarknet53


class YOLOv4(tf.keras.Model):
    """YOLOv4 Model"""
    def __init__(self, input_shape):
        """
        Constructor

        Args:
            input_shape (Tuple[int]): Input shape of the image
        """
        super(YOLOv4, self).__init__()
        self.cspdarknet53 = CSPDarknet53(input_shape)

    def call(self, inputs, training=None, mask=None):
        return self.cspdarknet53(inputs)


if __name__ == "__main__":
    model = YOLOv4((416, 416, 3))
    outputs = model.predict(tf.random.uniform((16, 416, 416, 3)))
    print(outputs[0].shape)
