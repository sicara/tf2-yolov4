def test_backbone_should_have_valid_output_shapes(cspdarknet53_416):
    output_1, output_2, output_3 = cspdarknet53_416.outputs
    assert output_1.shape.as_list() == [None, 52, 52, 256]
    assert output_2.shape.as_list() == [None, 26, 26, 512]
    assert output_3.shape.as_list() == [None, 13, 13, 1024]

