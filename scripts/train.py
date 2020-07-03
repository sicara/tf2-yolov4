"""
Training script for Pascal VOC using tf2-yolov4
"""
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tf2_yolov4.anchors import (
    YOLOV4_ANCHORS,
    YOLOV4_ANCHORS_MASKS,
    compute_normalized_anchors,
)
from tf2_yolov4.datasets import prepare_dataset
from tf2_yolov4.losses import YoloV3Loss
from tf2_yolov4.model import YOLOv4

INPUT_SHAPE = (608, 608, 3)


def launch_training(batch_size, weights_path, all_frozen_epoch_number, backbone_frozen_epoch_number, num_epochs, dataset_name="voc"):
    LOG_DIR = Path("./logs") / dataset_name / datetime.now().strftime("%m-%d-%Y %H:%M:%S")

    voc_dataset, infos = tfds.load(dataset_name, with_info=True, shuffle_files=True)

    ds_train, ds_test = voc_dataset["train"], voc_dataset["validation"]
    ds_train = prepare_dataset(
        ds_train,
        shape=INPUT_SHAPE,
        batch_size=batch_size,
        shuffle=True,
        apply_data_augmentation=True,
        transform_to_bbox_by_stage=True,
    )
    ds_test = prepare_dataset(
        ds_test,
        shape=INPUT_SHAPE,
        batch_size=batch_size,
        shuffle=False,
        apply_data_augmentation=False,
        transform_to_bbox_by_stage=True,
    )

    steps_per_epoch = infos.splits["train"].num_examples // batch_size
    validation_steps = infos.splits["validation"].num_examples // batch_size
    num_classes = infos.features["objects"]["label"].num_classes

    model = YOLOv4(
        input_shape=INPUT_SHAPE,
        anchors=YOLOV4_ANCHORS,
        num_classes=num_classes,
        training=True,
    )
    if weights_path is not None:
        model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
        print("Darknet weights loaded.")

    optimizer = tf.keras.optimizers.Adam(1e-4)
    normalized_anchors = compute_normalized_anchors(YOLOV4_ANCHORS, INPUT_SHAPE)
    loss = [
        YoloV3Loss(np.concatenate(list(normalized_anchors), axis=0)[mask])
        for mask in YOLOV4_ANCHORS_MASKS
    ]

    # Start training: 5 epochs with backbone + neck frozen
    for layer in (
        model.get_layer("CSPDarknet53").layers + model.get_layer("YOLOv4_neck").layers
    ):
        layer.trainable = False
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(
        ds_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_test,
        validation_steps=validation_steps,
        epochs=all_frozen_epoch_number,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
            tf.keras.callbacks.ModelCheckpoint(
                str(LOG_DIR / "yolov4_all_frozen.h5"),
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
            ),
        ],
    )

    # Keep training: 10 epochs with backbone frozen -- unfreeze neck
    for layer in model.get_layer("YOLOv4_neck").layers:
        layer.trainable = True
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(
        ds_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_test,
        validation_steps=validation_steps,
        epochs=backbone_frozen_epoch_number + all_frozen_epoch_number,
        initial_epoch=all_frozen_epoch_number,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
            tf.keras.callbacks.ModelCheckpoint(
                str(LOG_DIR / "yolov4_backbone_frozen.h5"),
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
            ),
        ],
    )

    # Final training
    for layer in model.get_layer("CSPDarknet53").layers:
        layer.trainable = True
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(
        ds_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_test,
        validation_steps=validation_steps,
        epochs=num_epochs,
        initial_epoch=all_frozen_epoch_number + backbone_frozen_epoch_number,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
            tf.keras.callbacks.ModelCheckpoint(
                str(LOG_DIR / "yolov4_full.h5"),
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(LOG_DIR / "yolov4_train_loss.h5"),
                save_best_only=True,
                save_weights_only=True,
                monitor="loss",
            ),
        ],
    )


@click.command()
@click.option("--batch_size", type=int, default=16, help="Size of mini-batch")
@click.option("--weights_path", type=click.Path(exists=True), default=None, help="Path to pretrained weights")
@click.option("--all_frozen_epoch_number", type=int, default=20, help="Number of epochs to perform with backbone and neck frozen")
@click.option("--backbone_frozen_epoch_number", type=int, default=10, help="Number of epochs to perform with backbone frozen")
@click.option("--num_epochs", type=int, default=50, help="Total number of epochs to perform")
@click.option("--dataset_name", type=str, default="voc", help="Dataset used during training. Refer to TensorFlow Datasets documentation for dataset names.")
def launch_training_command(batch_size, weights_path, all_frozen_epoch_number, backbone_frozen_epoch_number, num_epochs, dataset_name):
    launch_training(batch_size, weights_path, all_frozen_epoch_number, backbone_frozen_epoch_number, num_epochs, dataset_name)


if __name__ == "__main__":
    launch_training_command()
