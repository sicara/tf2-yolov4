# tf2-yolov4

> A TensorFlow 2.0 implementation of YOLOv4: Optimal Speed and Accuracy of Object Detection

![Python Versions](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-%23EBBD68.svg)
![Tensorflow Versions](https://img.shields.io/badge/TensorFlow-2.x-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Python package](https://github.com/sicara/tf2-yolov4/workflows/Python%20package/badge.svg?branch=master)](https://github.com/sicara/tf2-yolov4/actions?query=workflow%3A%22Python+package%22)

This implementation runs (for now) inference with the original Darknet weights from [AlexeyAB](https://www.github.com/AlexeyAB/darknet).
See the roadmap section to see what's next.

## Installation

To install this package, you can run:

```bash
pip install https://github.com/sicara/tf2-yolov4/archive/master.zip
pip install tensorflow
# Check that tf2_yolov4 is installed properly
python -c "from tf2_yolov4.model import YOLOv4; print(YOLOv4)"
```

Check the [detect script](https://github.com/sicara/tf2-yolov4/blob/master/scripts/detect.py) to run a prediction.

Requirements:

- MacOs >= 10.15 since tensorflow-addons is not available for older release of MacOs
- Python >= 3.6
- Compatible versions between TensorFlow and TensorFlow Addons: check the [compatibility matrix](https://github.com/tensorflow/addons#python-op-compatibility-matrix)

## Pretrained weights

To load the Darknet weights trained on COCO, you have to:

- get the weights (yolov4.weights) from [AlexeyAB/darknet](https://www.github.com/AlexeyAB/darknet)
- run `python tools/convert_darknet_weights.py PATH_TO/yolov4.weights`

TF weights should be saved as `yolov4.h5`.
For more information about the conversion script, run `python tools/convert_darknet_weights.py --help`.

## Roadmap

- [x] Inference
    - [x] CSPDarknet53 backbone with Mish activations
    - [x] SPP Neck
    - [x] YOLOv3 Head
    - [x] Load Darknet Weights
    - [x] Image loading and preprocessing
    - [x] YOLOv3 box postprocessing
    - [x] Handling non-square images
- [ ] Training
    - [ ] Training loop with YOLOv3 loss
    - [ ] CIoU loss
    - [ ] Cross mini-Batch Normalization
    - [ ] Self-adversarial Training
    - [ ] Mosaic Data Augmentation
    - [ ] DropBlock
- [ ] Enhancements
    - [ ] Automatic download of pretrained weights (like Keras applications)
