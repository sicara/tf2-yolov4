import numpy as np
import tensorflow as tf
from tf_image.application.augmentation_config import AugmentationConfig
from tf_image.application.tools import random_augmentations
from tf_image.core.convert_type_decorator import convert_type

from tf2_yolov4.anchors import (
    YOLOV4_ANCHORS,
    YOLOV4_ANCHORS_MASKS,
    compute_normalized_anchors,
)

config = AugmentationConfig()


BOUNDING_BOXES_FIXED_NUMBER = 60


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


def random_flip_right_with_bounding_boxes(images, bounding_boxes):
    apply_flip = tf.random.uniform(shape=[]) > 0.5
    if apply_flip:
        images = tf.image.flip_left_right(images)
        bounding_boxes = tf.stack(
            [
                1.0 - bounding_boxes[..., 2],
                bounding_boxes[..., 1],
                1.0 - bounding_boxes[..., 0],
                bounding_boxes[..., 3],
                bounding_boxes[..., 4],
            ],
            axis=-1,
        )

    return images, bounding_boxes


@convert_type
def augment_image(image, bounding_boxes):
    bboxes, labels = bounding_boxes[:, :-1], bounding_boxes[:, -1]
    image_aug, bboxes_aug = random_augmentations(image, config, bboxes=bboxes)
    return (image_aug, tf.concat([bboxes_aug, tf.expand_dims(labels, axis=-1)], axis=1))


def prepare_dataset(
    dataset,
    shape,
    batch_size,
    shuffle=True,
    apply_data_augmentation=True,
    transform_to_bbox_by_stage=True,
    pad_number_of_boxes=BOUNDING_BOXES_FIXED_NUMBER,
    anchors=YOLOV4_ANCHORS,
):
    normalized_anchors = compute_normalized_anchors(anchors, shape)
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
        lambda image, bounding_box: (
            tf.image.resize(image, shape[:2]) / 255.0,
            bounding_box,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if apply_data_augmentation:
        dataset = dataset.map(
            augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    dataset = dataset.map(
        lambda image, bounding_box: (
            tf.image.resize(image, shape[:2]) / 255.0,
            bounding_box,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(
        lambda image, bounding_boxes: (
            image,
            pad_bounding_boxes_to_fixed_number_of_bounding_boxes(
                bounding_boxes, pad_number=pad_number_of_boxes
            ),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)

    if transform_to_bbox_by_stage:
        dataset = dataset.map(
            lambda image, bounding_box_with_class: (
                image,
                transform_targets(  # Comes straight from https://github.com/zzh8829/yolov3-tf2/
                    bounding_box_with_class,
                    np.concatenate(
                        list(normalized_anchors), axis=0
                    ),  # Must concatenate because in zzh8829/yolov3-tf2, it's a list of anchors
                    YOLOV4_ANCHORS_MASKS,
                    shape[0],  # Assumes square input
                ),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset.repeat()
