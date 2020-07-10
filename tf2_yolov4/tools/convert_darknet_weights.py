"""
Script to convert yolov4.weights file from AlexeyAB/darknet to tensorflow weights.

Initial implementation comes from https://github.com/zzh8829/yolov3-tf2
"""
from pathlib import Path

import click
import numpy as np

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
from tf2_yolov4.tools.weights import load_darknet_weights_in_yolo

INPUT_SHAPE = (416, 416, 3)
NUM_CLASSES = 80


# pylint: disable=too-many-locals
@click.command()
@click.argument("darknet-weights-path", type=click.Path(exists=True))
@click.option(
    "--output-weights-path",
    "-o",
    default=Path(".") / "yolov4.h5",
    type=click.Path(),
    help="Output tensorflow weights filepath (*.h5)",
)
def convert_darknet_weights(darknet_weights_path, output_weights_path):
    """ Converts yolov4 darknet weights to tensorflow weights (.h5 file)

    Args:
        darknet_weights_path (str): Input darknet weights filepath (*.weights).
        output_weights_path (str): Output tensorflow weights filepath (*.h5).
    """
    model = YOLOv4(
        input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, anchors=YOLOV4_ANCHORS
    )
    # pylint: disable=E1101
    model.predict(np.random.random((1, *INPUT_SHAPE)))

    model = load_darknet_weights_in_yolo(model, darknet_weights_path)

    model.save(output_weights_path)


if __name__ == "__main__":
    convert_darknet_weights()
