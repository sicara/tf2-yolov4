# tf2-yolov4

> A TensorFlow 2.0 implementation of YOLOv4: Optimal Speed and Accuracy of Object Detection

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Python package](https://github.com/sicara/tf2-yolov4/workflows/Python%20package/badge.svg?branch=master)](https://github.com/sicara/tf2-yolov4/actions?query=workflow%3A%22Python+package%22)

## Requirements:
- MacOs >= 10.15 since tensorflow-addons is not available for older release of MacOs
- Python >= 3.6 and < 3.8 since tensorflow-addons is not available for every python distribution

## COCO Weights

To load COCO weights into this implementation, you have to:

- get the weights (yolov4.weights) from [AlexeyAB/darknet](https://www.github.com/AlexeyAB/darknet)
- run `python tools/weights_convert.py`

TF weights should be saved as `yolov4.h5`.

## Work in Progress

- [ ] Inference
    - [x] CSPDarknet53 backbone with Mish activations
    - [x] SPP Neck
    - [x] YOLOv3 Head
    - [x] Load Darknet Weights
    - [x] Image loading and preprocessing
    - [x] YOLOv3 box postprocessing
- [ ] Training
    - [ ] Training loop with YOLOv3 loss
    - [ ] CIoU loss
    - [ ] Cross mini-Batch Normalization
    - [ ] Self-adversarial Training
    - [ ] Mosaic Data Augmentation
    - [ ] DropBlock
