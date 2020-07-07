import click
import tensorflow as tf

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

HEIGHT, WIDTH = (640, 960)

TFLITE_MODEL_PATH = "yolov4.tflite"


@click.command()
@click.option("--num_classes", default=80, help="Number of classes")
@click.option(
    "--weights_path", default=None, help="Path to .h5 file with model weights"
)
def convert_tflite(num_classes, weights_path):
    model = YOLOv4(
        input_shape=(HEIGHT, WIDTH, 3),
        anchors=YOLOV4_ANCHORS,
        num_classes=num_classes,
        training=False,
        yolo_max_boxes=100,
        yolo_iou_threshold=0.4,
        yolo_score_threshold=0.1,
    )

    if weights_path:
        model.load_weights(weights_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    convert_tflite()
