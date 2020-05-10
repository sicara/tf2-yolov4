import tensorflow as tf


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


def neck(input_shapes):
    """
    Implements the neck of YOLOv4, including the SPP.

    Args:
        input_shapes (List[Tuple[int]]): List of 3 tuples, which are the output shapes of the backbone.
            For CSPDarknet53, those are: [(13, 13, 1024), (26, 26, 512), (52, 52, 256)] for a (416, 416) input.

    Returns:
        tf.keras.Model: Neck model
    """
    input_1 = tf.keras.Input(shape=input_shapes[0])
    input_2 = tf.keras.Input(shape=input_shapes[1])
    input_3 = tf.keras.Input(shape=input_shapes[2])

    x1 = conv_bn_leaky(input_1, filters=512, kernel_size=1, strides=1)
    x1 = conv_bn_leaky(x1, filters=1024, kernel_size=3, strides=1)
    x1 = conv_bn_leaky(x1, filters=512, kernel_size=1, strides=1)

    maxpool_1 = tf.keras.layers.MaxPool2D((5, 5), strides=1, padding="same")(x1)
    maxpool_2 = tf.keras.layers.MaxPool2D((9, 9), strides=1, padding="same")(x1)
    maxpool_3 = tf.keras.layers.MaxPool2D((13, 13), strides=1, padding="same")(x1)

    spp = tf.keras.layers.Concatenate()([maxpool_3, maxpool_2, maxpool_1, x1])

    x = conv_bn_leaky(spp, filters=512, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=1024, kernel_size=3, strides=1)
    output_1 = conv_bn_leaky(x, filters=512, kernel_size=1, strides=1)
    x = conv_bn_leaky(output_1, filters=256, kernel_size=1, strides=1)

    upsampled = tf.keras.layers.UpSampling2D()(x)

    x = conv_bn_leaky(input_2, filters=256, kernel_size=1, strides=1)
    x = tf.keras.layers.Concatenate()([x, upsampled])

    x = conv_bn_leaky(x, filters=256, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=512, kernel_size=3, strides=1)
    x = conv_bn_leaky(x, filters=256, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=512, kernel_size=3, strides=1)
    output_2 = conv_bn_leaky(x, filters=256, kernel_size=1, strides=1)
    x = conv_bn_leaky(output_2, filters=128, kernel_size=1, strides=1)

    upsampled = tf.keras.layers.UpSampling2D()(x)

    x = conv_bn_leaky(input_3, filters=128, kernel_size=1, strides=1)
    x = tf.keras.layers.Concatenate()([x, upsampled])

    x = conv_bn_leaky(x, filters=128, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=256, kernel_size=3, strides=1)
    x = conv_bn_leaky(x, filters=128, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=256, kernel_size=3, strides=1)
    output_3 = conv_bn_leaky(x, filters=128, kernel_size=1, strides=1)

    return tf.keras.Model([input_1, input_2, input_3], [output_3, output_2, output_1])


if __name__ == "__main__":
    model = neck([(13, 13, 1024), (26, 26, 512), (52, 52, 256)])
    model.summary()
