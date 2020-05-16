"""Common layer architecture such as Conv->BN->Mish or Conv->BN->LeakyReLU"""
import tensorflow as tf
import tensorflow_addons as tfa

def conv_bn_leaky(inputs, filters, kernel_size, strides, padding="same"):
    """
    Applies successively Conv2D -> BN -> LeakyReLU

    Args:
        inputs (tf.Tensor): 4D (N,H,W,C) input tensor
        filters (int): Number of convolutional filters
        kernel_size (int): Size of the convolutional kernel
        strides (int): Strides used for the convolution
        padding (str): Type of padding used in the convolution

    Returns:
        tf.Tensor: 4D (N,H,W,C) output tensor
    """
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    return x


def conv_bn_mish(inputs, filters, kernel_size, strides, padding="same"):
    """
    Applies successively Conv2D -> BN -> Mish

    Args:
        inputs (tf.Tensor): 4D (N,H,W,C) input tensor
        filters (int): Number of convolutional filters
        kernel_size (int): Size of the convolutional kernel
        strides (int): Strides used for the convolution
        padding (str): Type of padding used in the convolution

    Returns:
        tf.Tensor: 4D (N,H/strides,W/strides,filters) output tensor
    """
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tfa.activations.mish(x)

    return x
