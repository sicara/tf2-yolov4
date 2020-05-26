"""
Script to convert yolov4.weights file from AlexeyAB/darknet to TF2.x

Initial implementation comes from https://github.com/zzh8829/yolov3-tf2
"""
from pathlib import Path

import click
import numpy as np

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4


# pylint: disable=too-many-locals
@click.command()
@click.argument("darknet-weights-path", type=click.Path(exists=True))
@click.option(
    "--output-weights-path",
    "-o",
    default=Path(".") / "yolov4.h5",
    type=click.Path(),
    help="Output weights TF2.x filepath (*.h5)",
)
@click.option(
    "--input-shape",
    "-s",
    default=(416, 416, 3),
    type=click.Tuple([int, int, int]),
    help="Input shape of the image",
)
@click.option("--num-classes", "-n", default=80, type=int, help="Number of classes")
def convert_darknet_weights(
    darknet_weights_path, output_weights_path, input_shape, num_classes
):
    """ Converts yolov4 darknet weights to TF2.x weights (.h5 file)

    Args:
        darknet_weights_path (str): Input darknet weights filepath (*.weights).
        output_weights_path (str): Output weights TF2.x filepath (*.h5).
        input_shape (Tuple[int]): Input shape of the image.
        num_classes (int): Number of classes.
    """
    model = YOLOv4(
        input_shape=input_shape, num_classes=num_classes, anchors=YOLOV4_ANCHORS
    )
    # pylint: disable=E1101
    model.predict(np.random.random((1, *input_shape)))

    sample_conv_weights = (
        model.get_layer("CSPDarknet53").get_layer("conv2d_32").get_weights()[0]
    )

    model_layers = (
        model.get_layer("CSPDarknet53").layers
        + model.get_layer("YOLOv4_neck").layers
        + model.get_layer("YOLOv3_head").layers
    )

    # Get all trainable layers: convolutions and batch normalization
    conv_layers = [layer for layer in model_layers if "conv2d" in layer.name]
    batch_norm_layers = [
        layer for layer in model_layers if "batch_normalization" in layer.name
    ]

    # Sort them by order of appearance.
    # The first apparition does not have an index, hence the [[0]] trick to avoid an error
    conv_layers = [conv_layers[0]] + sorted(
        conv_layers[1:], key=lambda x: int(x.name[7:])
    )
    batch_norm_layers = [batch_norm_layers[0]] + sorted(
        batch_norm_layers[1:], key=lambda x: int(x.name[20:])
    )

    # Open darknet file and read headers
    darknet_weight_file = open(darknet_weights_path, "rb")
    # pylint: disable=unused-variable
    major, minor, revision, seen, _ = np.fromfile(
        darknet_weight_file, dtype=np.int32, count=5
    )

    # Keep an index of which batch norm should be considered.
    # If batch norm is used with a convolution (meaning conv has no bias), the index is incremented
    # Otherwise (conv has a bias), index is kept still.
    current_matching_batch_norm_index = 0

    for layer in conv_layers:
        kernel_size = layer.kernel_size
        input_filters = layer.input_shape[-1]
        filters = layer.filters
        use_bias = layer.bias is not None

        if use_bias:
            # Read bias weight
            conv_bias = np.fromfile(
                darknet_weight_file, dtype=np.float32, count=filters
            )
        else:
            # Read batch norm
            # Reorder from darknet (beta, gamma, mean, var) to TF (gamma, beta, mean, var)
            batch_norm_weights = np.fromfile(
                darknet_weight_file, dtype=np.float32, count=4 * filters
            ).reshape((4, filters))[[1, 0, 2, 3]]

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
            batch_norm_layers[current_matching_batch_norm_index].set_weights(
                batch_norm_weights
            )
            current_matching_batch_norm_index += 1

    #  Check if we read the entire darknet file.
    remaining_chars = len(darknet_weight_file.read())
    click.echo(
        f"Number of remaining values to load from darknet weights file: {remaining_chars}"
    )
    darknet_weight_file.close()
    assert remaining_chars == 0

    # Check if weights have been updated
    sample_conv_weights_after_loading = (
        model.get_layer("CSPDarknet53").get_layer("conv2d_32").get_weights()[0]
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        sample_conv_weights,
        sample_conv_weights_after_loading,
    )

    model.save(output_weights_path)


if __name__ == "__main__":
    convert_darknet_weights()
