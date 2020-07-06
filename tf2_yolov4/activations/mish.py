"""
Tensorflow-Keras Implementation of Mish
Source: https://github.com/digantamisra98/Mish/blob/master/Mish/TFKeras/mish.py
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Mish(Layer):
    """
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Mish()(X_input)
    """

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))
