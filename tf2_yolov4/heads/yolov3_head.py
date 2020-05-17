"""
Implements YOLOv3 head, which is used in YOLOv4

Implementation mainly inspired by from https://github.com/zzh8829/yolov3-tf2
"""
import tensorflow as tf

from tf2_yolov4.config.anchors import YOLOv4Config
from tf2_yolov4.layers import conv_bn_leaky


def yolov3_head(
    input_shapes,
    anchors,
    num_classes,
    predict_boxes,
    yolo_max_boxes,
    yolo_iou_threshold,
    yolo_score_threshold,
):
    """
    Returns the YOLOv3 head, which is used in YOLOv4

    Args:
        input_shapes (List[Tuple[int]]): List of 3 tuples, which are the output shapes of the neck.
            None dimensions are ignored.
            For CSPDarknet53+YOLOv4_neck, those are: [(13, 13, 512), (26, 26, 256) (52, 52, 128)] for a (416,416) input.
        anchors (List[numpy.array[int, 2]]): List of 3 numpy arrays containing the anchor sizes used for each stage.
            The first and second columns of the numpy arrays respectively contain the anchors height and width.
        num_classes (int): Number of classes.
        predict_boxes (boolean): If True, will output boxes computed through YOLO regression and NMS, and YOLO features
            otherwise. In most case, set it True for inference, and False for training.
        yolo_max_boxes (int): Maximum number of boxes predicted on each image (across all anchors/stages)
        yolo_iou_threshold (float between 0. and 1.): IOU threshold defining whether close boxes will be merged
            during non max regression.
        yolo_score_threshold (float between 0. and 1.): Boxes with score lower than this threshold will be filtered
            out during non max regression.
    Returns:
        tf.keras.Model: Head model
    """
    input_1 = tf.keras.Input(shape=filter(None, input_shapes[0]))
    input_2 = tf.keras.Input(shape=filter(None, input_shapes[1]))
    input_3 = tf.keras.Input(shape=filter(None, input_shapes[2]))

    x = conv_bn_leaky(input_3, filters=256, kernel_size=3, strides=1)
    output_3 = conv_classes_anchors(
        x, num_anchors_stage=len(anchors[0]), num_classes=num_classes
    )

    x = conv_bn_leaky(input_3, filters=256, kernel_size=3, strides=2)
    x = tf.keras.layers.Concatenate()([x, input_2])
    x = conv_bn_leaky(x, filters=256, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=512, kernel_size=3, strides=1)
    x = conv_bn_leaky(x, filters=256, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=512, kernel_size=3, strides=1)
    connection = conv_bn_leaky(x, filters=256, kernel_size=1, strides=1)
    x = conv_bn_leaky(connection, filters=512, kernel_size=3, strides=1)
    output_2 = conv_classes_anchors(
        x, num_anchors_stage=len(anchors[1]), num_classes=num_classes
    )

    x = conv_bn_leaky(connection, filters=512, kernel_size=3, strides=2)
    x = tf.keras.layers.Concatenate()([x, input_1])
    x = conv_bn_leaky(x, filters=512, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=1024, kernel_size=3, strides=1)
    x = conv_bn_leaky(x, filters=512, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=1024, kernel_size=3, strides=1)
    x = conv_bn_leaky(x, filters=512, kernel_size=1, strides=1)
    x = conv_bn_leaky(x, filters=1024, kernel_size=3, strides=1)
    output_1 = conv_classes_anchors(
        x, num_anchors_stage=len(anchors[2]), num_classes=num_classes
    )

    if not predict_boxes:
        return tf.keras.Model(
            [input_1, input_2, input_3],
            [output_1, output_2, output_3],
            name="YOLOv3_head",
        )

    boxes_1 = tf.keras.layers.Lambda(
        lambda x: yolov3_boxes_regression(x, anchors[0], num_classes),
        name="yolov3_boxes_regression_1",
    )(output_1)
    boxes_2 = tf.keras.layers.Lambda(
        lambda x: yolov3_boxes_regression(x, anchors[1], num_classes),
        name="yolov3_boxes_regression_2",
    )(output_2)
    boxes_3 = tf.keras.layers.Lambda(
        lambda x: yolov3_boxes_regression(x, anchors[2], num_classes),
        name="yolov3_boxes_regression_3",
    )(output_3)

    output = tf.keras.layers.Lambda(
        lambda x: yolo_nms(
            x,
            yolo_max_boxes=yolo_max_boxes,
            yolo_iou_threshold=yolo_iou_threshold,
            yolo_score_threshold=yolo_score_threshold,
        ),
        name="yolov4_nms",
    )([boxes_1[:3], boxes_2[:3], boxes_3[:3]])

    return tf.keras.Model([input_1, input_2, input_3], output, name="YOLOv4_head")


def conv_classes_anchors(inputs, num_anchors_stage, num_classes):
    """
    Applies Conv2D based on the number of anchors and classifications classes, then reshape the Tensor.
    TODO:  doubleCheck use_bias=True: is False in yolov3_tf2, is not specified in yolov4.cfg and True in pytorch yolov4.

    Args:
        inputs (tf.Tensor): 4D (N,H,W,C) input tensor
        num_anchors_stage (int): Number of anchors for the given output stage
        num_classes (int): Number of classes

    Returns:
        tf.Tensor: 5D (N,H,W,num_anchors_stage,num_classes+5) output tensor.
            The last dimension contains the 4 box coordinates regression factors, the 1 objectness score,
            and the num_classes confidence scores
    """
    x = tf.keras.layers.Conv2D(
        filters=num_anchors_stage * (num_classes + 5),
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=True,
    )(inputs)
    x = tf.keras.layers.Reshape(
        (x.shape[1], x.shape[2], num_anchors_stage, num_classes + 5)
    )(x)
    return x


def yolov3_boxes_regression(features, anchors_per_stage, num_classes):
    # pred: (batch_size, grid_x, grid_y, anchors, (x, y, w, h, obj, ...classes))
    grid_size_x, grid_size_y = features.shape[1], features.shape[2]
    box_xy, box_wh, objectness, class_probs = tf.split(
        features, (2, 2, 1, num_classes), axis=-1
    )

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    grid = tf.meshgrid(tf.range(grid_size_x), tf.range(grid_size_y), indexing="ij")
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.constant(
        [grid_size_x, grid_size_y], dtype=tf.float32
    )
    box_wh = tf.exp(box_wh) * anchors_per_stage

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(pred, yolo_max_boxes, yolo_iou_threshold, yolo_score_threshold):
    """


    Args:
        pred (List[Tuple[tf.Tensor]]): For each output stage, returns a 3-tuple of 5D tensors corresponding to
            bbox (N,grid_x,grid_y,anchor,4),
            objectness (N,grid_x,grid_y,anchor,4),
            class_probs (N,grid_x,grid_y,anchor,num_classes),
        yolo_max_boxes (int): Maximum number of boxes predicted on each image (across all anchors/stages)
        yolo_iou_threshold (float between 0. and 1.): IOU threshold defining whether close boxes will be merged
            during non max regression.
        yolo_score_threshold (float between 0. and 1.): Boxes with score lower than this threshold will be filtered
            out during non max regression.
    """
    bbox_per_stage, objectness_per_stage, class_probs_per_stage = [], [], []

    for stage_pred in pred:
        num_boxes = (
            stage_pred[0].shape[1] * stage_pred[0].shape[2] * stage_pred[0].shape[3]
        )  # num_anchors * grid_x * grid_y
        bbox_per_stage.append(
            tf.reshape(
                stage_pred[0],
                (tf.shape(stage_pred[0])[0], num_boxes, stage_pred[0].shape[-1]),
            )
        )  # [None,num_boxes,4]
        objectness_per_stage.append(
            tf.reshape(
                stage_pred[1],
                (tf.shape(stage_pred[1])[0], num_boxes, stage_pred[1].shape[-1]),
            )
        )  # [None,num_boxes,1]
        class_probs_per_stage.append(
            tf.reshape(
                stage_pred[2],
                (tf.shape(stage_pred[2])[0], num_boxes, stage_pred[2].shape[-1]),
            )
        )  # [None,num_boxes,num_classes]

    bbox = tf.concat(bbox_per_stage, axis=1)
    objectness = tf.concat(objectness_per_stage, axis=1)
    class_probs = tf.concat(class_probs_per_stage, axis=1)

    scores = objectness * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(bbox, axis=2),
        scores=scores,
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold,
    )

    return [boxes, scores, classes, valid_detections]


if __name__ == "__main__":

    model = yolov3_head(
        [(13, 13, 1024), (26, 26, 512), (52, 52, 256)],
        anchors=YOLOv4Config.get_yolov3_anchors(),
        num_classes=80,
        predict_boxes=False,
        yolo_max_boxes=50,
        yolo_iou_threshold=0.5,
        yolo_score_threshold=0.8,
    )
    model.summary()
