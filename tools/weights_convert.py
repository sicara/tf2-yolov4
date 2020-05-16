"""
Script to convert yolov4.weights file from AlexeyAB/darknet to TF2.x

Initial implementation comes from https://github.com/zzh8829/yolov3-tf2
"""
from pathlib import Path

import numpy as np

from tf2_yolov4.config.anchors import YOLOv4Config
from tf2_yolov4.model import YOLOv4


DARKNET_WEIGHTS_PATH = Path(".") / "yolov4.weights"

model = YOLOv4((416, 416, 3), YOLOv4Config.get_yolov4_anchors(), num_classes=80)
model.predict(np.random.random((1, 416, 416, 3)))

sample_conv_weights = model.get_layer("CSPDarknet53").get_layer("conv2d_32").get_weights()[0]

model_layers = model.get_layer("CSPDarknet53").layers + model.get_layer("YOLOv4_neck").layers + model.get_layer("YOLOv3_head").layers

# Get all trainable layers: convolutions and batch normalization
conv_layers = [layer for layer in model_layers if "conv2d" in layer.name]
batch_norm_layers = [layer for layer in model_layers if "batch_normalization" in layer.name]

# Sort them by order of appearance.
# The first apparition does not have an index, hence the [[0]] trick to avoid an error
conv_layers = [conv_layers[0]] + sorted(conv_layers[1:], key=lambda x: int(x.name[7:]))
batch_norm_layers = [batch_norm_layers[0]] + sorted(batch_norm_layers[1:], key=lambda x: int(x.name[20:]))

# Open darknet file and read headers
darknet_weight_file = open(DARKNET_WEIGHTS_PATH, "rb")
major, minor, revision, seen, _ = np.fromfile(darknet_weight_file, dtype=np.int32, count=5)

# Keep an index of which batch norm should be considered.
# If batch norm is used with a convolution (meaning conv has no bias), the index is incremented
# Otherwise (conv has a bias), index is kept still.
current_matching_batch_norm_index = 0

for layer_index, layer in enumerate(conv_layers):
    kernel_size = layer.kernel_size
    input_filters = layer.input_shape[-1]
    filters = layer.filters
    use_bias = layer.bias is not None

    if use_bias:
        # Read bias weight
        conv_bias = np.fromfile(darknet_weight_file, dtype=np.float32, count=filters)
    else:
        # Read batch norm
        # Reorder from darknet (beta, gamma, mean, var) to TF (gamma, beta, mean, var)
        batch_norm_weights = (
            np.fromfile(darknet_weight_file, dtype=np.float32, count=4 * filters)
              .reshape((4, filters))[[1, 0, 2, 3]]
        )

    # Read kernel weights
    # Reorder from darknet (filters, input_filters, kernel_size[0], kernel_size[1]) to
    # TF (kernel_size[0], kernel_size[1], input_filters, filters)
    conv_size = kernel_size[0] * kernel_size[1] * input_filters * filters
    conv_weights = (
        np.fromfile(darknet_weight_file, dtype=np.float32, count=conv_size)
          .reshape((filters, input_filters, kernel_size[0], kernel_size[1]))
          .transpose([2, 3, 1, 0])
    )

    if use_bias:
        # load conv weights and bias
        # increase batch_norm offset
        layer.set_weights([conv_weights, conv_bias])
    else:
        # load conv weights
        # load batch norm weights
        layer.set_weights([conv_weights])
        batch_norm_layers[current_matching_batch_norm_index].set_weights(batch_norm_weights)
        current_matching_batch_norm_index += 1

# Check if we read the entire darknet file.
remaining_chars = len(darknet_weight_file.read())
print(f"Number of remaining values to load from darknet weights file: {remaining_chars}")
darknet_weight_file.close()
assert remaining_chars == 0

# Check if weights have been updated
sample_conv_weights_after_loading = model.get_layer("CSPDarknet53").get_layer("conv2d_32").get_weights()[0]
np.testing.assert_raises(
    AssertionError,
    np.testing.assert_array_equal,
    sample_conv_weights,
    sample_conv_weights_after_loading,
)

model.save("yolov4.h5")
