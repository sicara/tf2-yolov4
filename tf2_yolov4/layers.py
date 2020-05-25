"""Common layer architecture such as Conv->BN->Mish or Conv->BN->LeakyReLU"""
import tensorflow as tf
import tensorflow_addons as tfa


def conv_bn(
    inputs,
    filters,
    kernel_size,
    strides,
    padding="same",
    zero_pad=False,
    activation="leaky",
):
    """
    Applies successively Conv2D -> BN -> LeakyReLU

    Args:
        inputs (tf.Tensor): 4D (N,H,W,C) input tensor
        filters (int): Number of convolutional filters
        kernel_size (int): Size of the convolutional kernel
        strides (int): Strides used for the convolution
        padding (str): Type of padding used in the convolution
        zero_pad (bool): If true, will zero-pad the input
        activation (string): Activation layer. Can be "mish" or "leaky_relu", or linear otherwise

    Returns:
        tf.Tensor: 4D (N,H,W,C) output tensor
    """
    if zero_pad:
        inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation == "leaky_relu":
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    elif activation == "mish":
        x = tfa.activations.mish(x)

    return x
