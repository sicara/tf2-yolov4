"""Implements YOLOv3 head, which is used in YOLOv4"""
import tensorflow as tf

from tf2_yolov4.config.anchors import YOLOv4Config
from tf2_yolov4.layers import conv_bn_leaky


def yolov3_head(input_shapes, anchors, num_classes):
    """
    Returns the YOLOv3 head, which is used in YOLOv4

    Args:
        input_shapes (List[Tuple[int]]): List of 3 tuples, which are the output shapes of the neck.
            None dimensions are ignored.
            For CSPDarknet53+YOLOv4_neck, those are: [(13, 13, 512), (26, 26, 256) (52, 52, 128)] for a (416,416) input.
        anchors (List[numpy.array[int, 2]]): List of 3 numpy arrays containing the anchor sizes used for each stage.
            The first and second columns of the numpy arrays respectively contain the anchors height and width.
        num_classes (int): Number of classes.

    Returns:
        tf.keras.Model: Head model
    """
    input_1 = tf.keras.Input(shape=filter(None, input_shapes[0]))
    input_2 = tf.keras.Input(shape=filter(None, input_shapes[1]))
    input_3 = tf.keras.Input(shape=filter(None, input_shapes[2]))

    x = conv_bn_leaky(input_3, filters=256, kernel_size=3, strides=1)
    output_3 = conv_classes_anchors(
        x, num_anchors_stage=len(anchors[0]), num_classes=num_classes
    )

    x = conv_bn_leaky(input_3, filters=256, kernel_size=3, strides=2)
    x = tf.keras.layers.Concatenate()([x, input_2])
    x = conv_bn_leaky(x, filters=256, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=512, kernel_size=3, strides=1)
    x = conv_bn_leaky(x, filters=256, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=512, kernel_size=3, strides=1)
    connection = conv_bn_leaky(x, filters=256, kernel_size=1, strides=1)
    x = conv_bn_leaky(connection, filters=512, kernel_size=3, strides=1)
    output_2 = conv_classes_anchors(
        x, num_anchors_stage=len(anchors[1]), num_classes=num_classes
    )

    x = conv_bn_leaky(connection, filters=512, kernel_size=3, strides=2)
    x = tf.keras.layers.Concatenate()([x, input_1])
    x = conv_bn_leaky(x, filters=512, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=1024, kernel_size=3, strides=1)
    x = conv_bn_leaky(x, filters=512, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=1024, kernel_size=3, strides=1)
    x = conv_bn_leaky(x, filters=512, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=1024, kernel_size=3, strides=1)
    output_1 = conv_classes_anchors(
        x, num_anchors_stage=len(anchors[2]), num_classes=num_classes
    )

    return tf.keras.Model(
        [input_1, input_2, input_3], [output_1, output_2, output_3], name="YOLOv3_head"
    )


def conv_classes_anchors(inputs, num_anchors_stage, num_classes):
    """
    Applies Conv2D based on the number of anchors and classifications classes, then reshape the Tensor.
    TODO:  doubleCheck use_bias=True: is False in yolov3_tf2, is not specified in yolov4.cfg and True in pytorch yolov4.

    Args:
        inputs (tf.Tensor): 4D (N,H,W,C) input tensor
        num_anchors_stage (int): Number of anchors for the given output stage
        num_classes (int): Number of classes

    Returns:
        tf.Tensor: 5D (N,H,W,num_anchors_stage,num_classes+5) output tensor.
            The last dimension contains the 4 box coordinates regression factors, the 1 objectness score,
            and the num_classes confidence scores
    """
    x = tf.keras.layers.Conv2D(
        filters=num_anchors_stage * (num_classes + 5),
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=True,
    )(inputs)
    x = tf.keras.layers.Reshape(
        (x.shape[1], x.shape[2], num_anchors_stage, num_classes + 5)
    )(x)
    return x


if __name__ == "__main__":

    model = yolov3_head(
        [(13, 13, 1024), (26, 26, 512), (52, 52, 256)],
        anchors=YOLOv4Config.get_yolov3_anchors(),
        num_classes=80,
    )
    model.summary()
