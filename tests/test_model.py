import tensorflow as tf


def test_model_should_predict_valid_shapes(yolov4):
    n_images = 2
    output_1, output_2, output_3 = yolov4.predict(tf.random.uniform((n_images, 416, 416, 3)))

    assert output_1.shape == (n_images, 13, 13, 3, 85)
    assert output_2.shape == (n_images, 26, 26, 3, 85)
    assert output_3.shape == (n_images, 52, 52, 3, 85)
