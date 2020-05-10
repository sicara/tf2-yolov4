# tf2-yolov4

> A TensorFlow 2.0 implementation of YOLOv4: Optimal Speed and Accuracy of Object Detection

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Python package](https://github.com/sicara/tf2-yolov4/workflows/Python%20package/badge.svg?branch=master)](https://github.com/sicara/tf2-yolov4/actions?query=workflow%3A%22Python+package%22)

## Requirements:
- MacOs >= 10.15 since tensorflow-addons is not available for older release of MacOs
- Python >= 3.5 and < 3.8 since tensorflow-addons is not available for every python distribution

## Work in Progress

- [ ] Inference
    - [x] CSPDarknet53 backbone with Mish activations
    - [x] SPP Neck
    - [ ] YOLOv3 Head
    - [ ] Load Darknet Weights
- [ ] Training
    - [ ] Training loop with YOLOv3 loss
    - [ ] CIoU loss
    - [ ] Cross mini-Batch Normalization
    - [ ] Self-adversarial Training
    - [ ] Mosaic Data Augmentation
    - [ ] DropBlock
