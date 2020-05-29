from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="tf2_yolov4",
    version="0.1.0",
    description="TensorFlow 2.x implementation of YOLOv4",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sicara/tf2-yolov4",
    license="MIT",
    install_requires=["click>=6.7", "numpy>=1.10", "tensorflow-addons>=0.9.1"],
    extras_require={"publish": ["bumpversion>=0.5.3", "twine>=1.13.0"]},
    packages=find_packages(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "convert-darknet-weights = tf2_yolov4.tools.convert_darknet_weights:convert_darknet_weights"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)
