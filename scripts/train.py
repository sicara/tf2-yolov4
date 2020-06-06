"""
Training script for Pascal VOC using tf2-yolov4
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tf2_yolov4.anchors import YOLOV4_ANCHORS, compute_normalized_anchors
from tf2_yolov4.heads.yolov3_head import yolov3_boxes_regression
from tf2_yolov4.model import YOLOv4

INPUT_SHAPE = (608, 608, 3)
BATCH_SIZE = 8
BOUNDING_BOXES_FIXED_NUMBER = 50
PASCAL_VOC_NUM_CLASSES = 20

YOLOV4_ANCHORS_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
YOLOV4_ANCHORS_NORMALIZED = compute_normalized_anchors(YOLOV4_ANCHORS, INPUT_SHAPE)

LOG_DIR = Path("./logs") / datetime.now().strftime("%m-%d-%Y %H:%M:%S")

ALL_FROZEN_EPOCH_NUMBER = 10
BACKBONE_FROZEN_EPOCH_NUMBER = 10
TOTAL_NUMBER_OF_EPOCHS = 50


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(
        tf.minimum(box_1[..., 2], box_2[..., 2])
        - tf.maximum(box_1[..., 0], box_2[..., 0]),
        0,
    )
    int_h = tf.maximum(
        tf.minimum(box_1[..., 3], box_2[..., 3])
        - tf.maximum(box_1[..., 1], box_2[..., 1]),
        0,
    )
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def YoloLoss(anchors, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolov3_boxes_regression(
            y_pred, anchors
        )
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(
                broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))),
                axis=-1,
            ),
            (pred_box, true_box, obj_mask),
            tf.float32,
        )
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = (
            obj_mask
            * box_loss_scale
            * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        )
        wh_loss = (
            obj_mask
            * box_loss_scale
            * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        )
        obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(
            true_class_idx, pred_class
        )

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]]
                )
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]]
                )
                idx += 1

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(
        box_wh[..., 1], anchors[..., 1]
    )
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def pad_bounding_boxes_to_fixed_number_of_bounding_boxes(bounding_boxes, pad_number):
    box_number = tf.shape(bounding_boxes)[0]
    paddings = [[0, pad_number - box_number], [0, 0]]

    return tf.pad(bounding_boxes, paddings, constant_values=0.0)


def prepare_dataset(dataset, shuffle=True):
    dataset = dataset.map(lambda el: (el["image"], el["objects"]))
    dataset = dataset.map(
        lambda image, object: (
            image,
            tf.concat(
                [
                    tf.stack(
                        [
                            object["bbox"][:, 1],
                            object["bbox"][:, 0],
                            object["bbox"][:, 3],
                            object["bbox"][:, 2],
                        ],
                        axis=-1,
                    ),
                    tf.expand_dims(tf.cast(object["label"], tf.float32), axis=-1),
                ],
                axis=-1,
            ),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda image, bounding_boxes: (
            image,
            pad_bounding_boxes_to_fixed_number_of_bounding_boxes(
                bounding_boxes, pad_number=BOUNDING_BOXES_FIXED_NUMBER
            ),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda image, bounding_box: (
            tf.image.resize(image, INPUT_SHAPE[:2]) / 255.0,
            bounding_box,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(
        lambda image, bounding_box_with_class: (
            image,
            transform_targets(  # Comes straight from https://github.com/zzh8829/yolov3-tf2/
                bounding_box_with_class,
                np.concatenate(
                    list(reversed(YOLOV4_ANCHORS_NORMALIZED)), axis=0
                ),  # Must concatenate because in zzh8829/yolov3-tf2, it's a list of anchors
                YOLOV4_ANCHORS_MASKS,
                INPUT_SHAPE[0],  # Assumes square input
            ),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    return dataset


if __name__ == "__main__":
    voc_dataset = tfds.load("voc", shuffle_files=True)
    ds_train, ds_test = voc_dataset["train"], voc_dataset["test"]
    ds_train = prepare_dataset(ds_train, shuffle=True)
    ds_test = prepare_dataset(ds_test, shuffle=False)

    model = YOLOv4(
        input_shape=INPUT_SHAPE,
        anchors=YOLOV4_ANCHORS,
        num_classes=PASCAL_VOC_NUM_CLASSES,
        training=True,
    )
    darknet_weights = Path("./yolov4.h5")
    if darknet_weights.exists():
        model.load_weights(str(darknet_weights), by_name=True, skip_mismatch=True)
        print("Darknet weights loaded.")

    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss = [
        YoloLoss(
            np.concatenate(list(reversed(YOLOV4_ANCHORS_NORMALIZED)), axis=0)[mask]
        )
        for mask in YOLOV4_ANCHORS_MASKS
    ]

    model.summary()
    # Start training: 5 epochs with backbone + neck frozen
    for layer in (
        model.get_layer("CSPDarknet53").layers + model.get_layer("YOLOv4_neck").layers
    ):
        layer.trainable = False
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(
        ds_train,
        validation_data=ds_test,
        validation_steps=10,
        epochs=ALL_FROZEN_EPOCH_NUMBER,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
            tf.keras.callbacks.ModelCheckpoint(
                "yolov4_all_frozen.h5", save_best_only=True, save_weights_only=True
            ),
        ],
    )
    # Keep training: 10 epochs with backbone frozen -- unfreeze neck
    for layer in model.get_layer("YOLOv4_neck").layers:
        layer.trainable = True
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(
        ds_train,
        validation_data=ds_test,
        validation_steps=10,
        epochs=BACKBONE_FROZEN_EPOCH_NUMBER + ALL_FROZEN_EPOCH_NUMBER,
        initial_epoch=ALL_FROZEN_EPOCH_NUMBER,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
            tf.keras.callbacks.ModelCheckpoint(
                "yolov4_backbone_frozen.h5",
                save_best_only=True,
                save_weights_only=True,
                verbose=True,
            ),
        ],
    )
    # Final training: 35 epochs with all weights unfrozen
    for layer in model.get_layer("CSPDarknet53").layers:
        layer.trainable = True
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(
        ds_train,
        validation_data=ds_test,
        validation_steps=10,
        epochs=TOTAL_NUMBER_OF_EPOCHS,
        initial_epoch=ALL_FROZEN_EPOCH_NUMBER + BACKBONE_FROZEN_EPOCH_NUMBER,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
            tf.keras.callbacks.ModelCheckpoint(
                "yolov4_full.h5", save_best_only=True, save_weights_only=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "yolov4_train_loss.h5",
                save_best_only=True,
                save_weights_only=True,
                monitor="loss",
            ),
        ],
    )
