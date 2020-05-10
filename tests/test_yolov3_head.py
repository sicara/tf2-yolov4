def test_backbone_should_have_valid_output_shapes(yolov3_head_416):
    output_1, output_2, output_3 = yolov3_head_416.outputs
    assert output_1.shape.as_list() == [None, 13, 13, 3, 85]
    assert output_2.shape.as_list() == [None, 26, 26, 3, 85]
    assert output_3.shape.as_list() == [None, 52, 52, 3, 85]
