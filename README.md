# YOLOv4

> A TensorFlow 2.0 implementation of YOLOv4: Optimal Speed and Accuracy of Object Detection

[![Pypi Version](https://img.shields.io/pypi/v/tf2-yolov4.svg)](https://pypi.org/project/tf2-yolov4/)
![Python Versions](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-%23EBBD68.svg)
![Tensorflow Versions](https://img.shields.io/badge/TensorFlow-2.x-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Python package](https://github.com/sicara/tf2-yolov4/workflows/Python%20package/badge.svg?branch=master)](https://github.com/sicara/tf2-yolov4/actions?query=workflow%3A%22Python+package%22)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/tf2-yolov4/blob/master/notebooks/YoloV4_Dectection_Example.ipynb)

This implementation runs (for now) inference with the original Darknet weights from [AlexeyAB](https://www.github.com/AlexeyAB/darknet).
See the roadmap section to see what's next.

<p align="center">
    <img src="./assets/banner.jpeg" width="940" />
</p>

## Installation

To install this package, you can run:

```bash
pip install tf2_yolov4
pip install tensorflow
# Check that tf2_yolov4 is installed properly
python -c "from tf2_yolov4.model import YOLOv4; print(YOLOv4)"
```

Requirements:

- MacOs >= 10.15 since tensorflow-addons is not available for older release of MacOs
- Python >= 3.6
- Compatible versions between TensorFlow and TensorFlow Addons: check the [compatibility matrix](https://github.com/tensorflow/addons#python-op-compatibility-matrix)

## Examples in Colab

- [Run detection on a single image](./notebooks/YoloV4_Dectection_Example.ipynb) / [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/tf2-yolov4/blob/master/notebooks/YoloV4_Dectection_Example.ipynb)

## Pretrained weights

Our YOLOv4 implementation supports the `weights` argument similarly to Keras applications. To load a model with pretrained
weights, you can simply call:

```python
# Loads Darknet weights trained on COCO
model = YOLOv4(
    input_shape,
    num_classes,
    anchors,
    weights="darknet",
)
```

If weights are available locally, they will be used. Otherwise, they will be automatically downloaded.

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
    - [x] Automatic download of pretrained weights (like Keras applications)

## References

- [yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)
