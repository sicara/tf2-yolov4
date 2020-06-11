def test_head_should_have_valid_output_shapes_training(
    yolov3_head_416_training, num_classes
):
    bounding_box_shape = 4
    objectness_score_shape = 1
    expected_head_shape = (num_classes + objectness_score_shape) + bounding_box_shape

    output_1, output_2, output_3 = yolov3_head_416_training.outputs
    assert output_1.shape.as_list() == [None, 52, 52, 3, expected_head_shape]
    assert output_2.shape.as_list() == [None, 26, 26, 3, expected_head_shape]
    assert output_3.shape.as_list() == [None, 13, 13, 3, expected_head_shape]


def test_head_should_have_valid_output_shapes_inference(
    yolov3_head_416_inference, yolo_max_boxes
):
    boxes, scores, classes, valid_detections = yolov3_head_416_inference.outputs
    assert boxes.shape.as_list() == [None, yolo_max_boxes, 4]
    assert scores.shape.as_list() == [None, yolo_max_boxes]
    assert classes.shape.as_list() == [None, yolo_max_boxes]
    assert valid_detections.shape.as_list() == [None]
