import numpy as np


class YOLOv4Config:
    @staticmethod
    def get_yolov4_anchors():
        return [
            np.array([(142, 110), (192, 243), (459, 401)], np.float32) / 416,
            np.array([(36, 75), (76, 55), (72, 146)], np.float32) / 416,
            np.array([(12, 16), (19, 36), (40, 28)], np.float32) / 416,
        ]

    @staticmethod
    def get_yolov3_anchors():
        return [
            np.array([(116, 90), (156, 198), (373, 326)], np.float32) / 416,
            np.array([(30, 61), (62, 45), (59, 119)], np.float32) / 416,
            np.array([(10, 13), (16, 30), (33, 23)], np.float32) / 416,
        ]
