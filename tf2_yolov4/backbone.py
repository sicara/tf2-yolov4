import tensorflow as tf
import tensorflow_addons as tfa


def conv_bn_mish(x, filters, kernel_size, strides, padding="same"):
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tfa.activations.mish(x)

    return x


def residual_block(x, filters, num_blocks):
    for _ in range(num_blocks):
        block_inputs = x
        x = conv_bn_mish(x, filters, kernel_size=(1, 1), strides=1)
        x = conv_bn_mish(x, filters, kernel_size=(3, 3), strides=1)

        x = x + block_inputs

    return x


def CSPBlock(x, filters, num_blocks):
    half_filters = filters // 2

    x = conv_bn_mish(x, filters=filters, kernel_size=3, strides=2)
    route = conv_bn_mish(x, filters=half_filters, kernel_size=1, strides=1)
    x = conv_bn_mish(x, filters=half_filters, kernel_size=1, strides=1)

    x = residual_block(x, filters=filters // 2, num_blocks=num_blocks, shortcut=True)
    x = conv_bn_mish(x, filters=half_filters, kernel_size=1, strides=1)
    x = tf.keras.layers.Concatenate()([x, route])

    x = conv_bn_mish(x, filters=filters, kernel_size=1, strides=1)

    return x


def CSPDarknet53(input_shape):
    """
    CSPDarknet53 implementation based on AlexeyAB/darknet config

    https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
    """
    inputs = tf.keras.Input(shape=input_shape)

    # First downsampling: L29 -> L103
    x = conv_bn_mish(inputs, filters=32, kernel_size=3, strides=1)

    # This block could be expressed as a CSPBlock with modification of num_filters in the middle
    # For readability purpose, we chose to keep the CSPBlock as simple as possible and have a little redondancy
    x = conv_bn_mish(x, filters=64, kernel_size=3, strides=2)
    route = conv_bn_mish(x, filters=64, kernel_size=1, strides=1)

    shortcut = conv_bn_mish(x, filters=64, kernel_size=1, strides=1)
    x = conv_bn_mish(shortcut, filters=32, kernel_size=1, strides=1)
    x = conv_bn_mish(x, filters=64, kernel_size=3, strides=1)

    x = x + shortcut
    x = conv_bn_mish(x, filters=64, kernel_size=1, strides=1)
    x = tf.keras.layers.Concatenate()([x, route])
    x = conv_bn_mish(x, filters=64, kernel_size=1, strides=1)

    # Second downsampling: L105 -> L191
    x = CSPBlock(x, filters=128, num_blocks=2)

    # Third downsampling: L193 -> L400
    output_1 = CSPBlock(x, filters=256, num_blocks=8)

    # Fourth downsampling: L402 -> L614
    output_2 = CSPBlock(output_1, filters=512, num_blocks=8)

    # Fifth downsampling: L616 -> L744
    output_3 = CSPBlock(output_2, filters=1024, num_blocks=4)

    return tf.keras.Model(inputs, [output_1, output_2, output_3])


if __name__ == "__main__":
    cspdarknet53 = CSPDarknet53((416, 416, 3))
    cspdarknet53.summary()
