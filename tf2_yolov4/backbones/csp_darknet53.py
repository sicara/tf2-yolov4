"""Implements YOLOv4 backbone layer: CSPDarknet53"""
import tensorflow as tf

from tf2_yolov4.layers import conv_bn


def residual_block(inputs, num_blocks):
    """
    Applies several residual connections.

    Args:
        inputs (tf.Tensor): 4D (N,H,W,C) input tensor
        num_blocks (int): Number of residual blocks

    Returns:
        tf.Tensor: 4D (N,H,W,C) output Tensor
    """
    _, _, _, filters = inputs.shape
    x = inputs
    for _ in range(num_blocks):
        block_inputs = x
        x = conv_bn(x, filters, kernel_size=1, strides=1, activation="mish")
        x = conv_bn(x, filters, kernel_size=3, strides=1, activation="mish")

        x = x + block_inputs

    return x


def csp_block(inputs, filters, num_blocks):
    """
    Create a CSPBlock which applies the following scheme to the input (N, H, W, C):
        - the first part (N, H, W, C // 2) goes into a series of residual connection
        - the second part is directly concatenated to the output of the previous operation

    Args:
        inputs (tf.Tensor): 4D (N,H,W,C) input tensor
        filters (int): Number of filters to use
        num_blocks (int): Number of residual blocks to apply

    Returns:
        tf.Tensor: 4D (N,H/2,W/2,filters) output tensor
    """
    half_filters = filters // 2

    x = conv_bn(
        inputs,
        filters=filters,
        kernel_size=3,
        strides=2,
        zero_pad=True,
        padding="valid",
        activation="mish",
    )
    route = conv_bn(
        x, filters=half_filters, kernel_size=1, strides=1, activation="mish"
    )
    x = conv_bn(x, filters=half_filters, kernel_size=1, strides=1, activation="mish")

    x = residual_block(x, num_blocks=num_blocks)
    x = conv_bn(x, filters=half_filters, kernel_size=1, strides=1, activation="mish")
    x = tf.keras.layers.Concatenate()([x, route])

    x = conv_bn(x, filters=filters, kernel_size=1, strides=1, activation="mish")

    return x


def csp_darknet53(input_shape):
    """
    CSPDarknet53 implementation based on AlexeyAB/darknet config

    https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
    """
    inputs = tf.keras.Input(shape=input_shape)

    # First downsampling: L29 -> L103
    x = conv_bn(inputs, filters=32, kernel_size=3, strides=1, activation="mish")

    # This block could be expressed as a CSPBlock with modification of num_filters in the middle
    # For readability purpose, we chose to keep the CSPBlock as simple as possible and have a little redondancy
    x = conv_bn(
        x,
        filters=64,
        kernel_size=3,
        strides=2,
        zero_pad=True,
        padding="valid",
        activation="mish",
    )
    route = conv_bn(x, filters=64, kernel_size=1, strides=1, activation="mish")

    shortcut = conv_bn(x, filters=64, kernel_size=1, strides=1, activation="mish")
    x = conv_bn(shortcut, filters=32, kernel_size=1, strides=1, activation="mish")
    x = conv_bn(x, filters=64, kernel_size=3, strides=1, activation="mish")

    x = x + shortcut
    x = conv_bn(x, filters=64, kernel_size=1, strides=1, activation="mish")
    x = tf.keras.layers.Concatenate()([x, route])
    x = conv_bn(x, filters=64, kernel_size=1, strides=1, activation="mish")

    # Second downsampling: L105 -> L191
    x = csp_block(x, filters=128, num_blocks=2)

    # Third downsampling: L193 -> L400
    output_1 = csp_block(x, filters=256, num_blocks=8)

    # Fourth downsampling: L402 -> L614
    output_2 = csp_block(output_1, filters=512, num_blocks=8)

    # Fifth downsampling: L616 -> L744
    output_3 = csp_block(output_2, filters=1024, num_blocks=4)

    return tf.keras.Model(inputs, [output_1, output_2, output_3], name="CSPDarknet53")
