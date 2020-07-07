from tf2_yolov4.tools.convert_tflite import create_tflite_model

HEIGHT, WIDTH = (640, 960)


def test_import_convert_tflite_script_does_not_fail():
    from tf2_yolov4.tools.convert_tflite import convert_tflite


def test_create_tflite_model_returns_correct_type(yolov4_inference):
    tflite_model = create_tflite_model(yolov4_inference)
    assert isinstance(tflite_model, bytes)
