def test_neck_should_have_valid_output_shapes(yolov4_neck_416):
    output_1, output_2, output_3 = yolov4_neck_416.outputs
    assert output_1.shape.as_list() == [None, 13, 13, 512]
    assert output_2.shape.as_list() == [None, 26, 26, 256]
    assert output_3.shape.as_list() == [None, 52, 52, 128]
