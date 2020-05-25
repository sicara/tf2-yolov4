import tensorflow as tf

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

SIZE_X, SIZE_Y = (480, 608)

#%% Load image
image = tf.io.read_file("scripts/images/cars.jpg")
image = tf.image.decode_image(image)
image = tf.expand_dims(image, axis=0)
image = tf.image.resize(image, (SIZE_X, SIZE_Y))
images = image / 255.0

#%% Load model
model = YOLOv4(
    input_shape=(SIZE_X, SIZE_Y, 3),
    anchors=YOLOV4_ANCHORS,
    num_classes=80,
    training=False,
    yolo_max_boxes=15,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.5,
)
model.load_weights("tools/yolov4.h5")

#%% Predict
predictions = model.predict(images)

#%% Printing boxes
boxes_x1y1_x2y2 = predictions[0]
boxes_y1x1_y2x2 = boxes_x1y1_x2y2[:, :, [1, 0, 3, 2]]
bbox_image = tf.image.draw_bounding_boxes(images, boxes_y1x1_y2x2, colors=None)
bbox_image = tf.cast(bbox_image * 255, "uint8")
tf.io.write_file(
    "scripts/output/detection.jpg", tf.io.encode_jpeg(bbox_image[0, :, :, :])
)
