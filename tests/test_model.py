import tensorflow as tf


def test_model_should_predict_valid_shapes(yolov4, num_classes):
    n_images = 2

    bounding_box_shape = 4
    objectness_score_shape = 1
    expected_head_shape = (num_classes + objectness_score_shape) + bounding_box_shape

    output_1, output_2, output_3 = yolov4.predict(tf.random.uniform((n_images, 416, 416, 3)))

    assert output_1.shape == (n_images, 13, 13, 3, expected_head_shape)
    assert output_2.shape == (n_images, 26, 26, 3, expected_head_shape)
    assert output_3.shape == (n_images, 52, 52, 3, expected_head_shape)
