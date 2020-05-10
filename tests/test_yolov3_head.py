def test_head_should_have_valid_output_shapes(yolov3_head_416, num_classes):
    bounding_box_shape = 4
    objectness_score_shape = 1
    expected_head_shape = (num_classes + objectness_score_shape) + bounding_box_shape

    output_1, output_2, output_3 = yolov3_head_416.outputs
    assert output_1.shape.as_list() == [None, 13, 13, 3, expected_head_shape]
    assert output_2.shape.as_list() == [None, 26, 26, 3, expected_head_shape]
    assert output_3.shape.as_list() == [None, 52, 52, 3, expected_head_shape]
