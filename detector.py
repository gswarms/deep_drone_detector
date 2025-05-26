""" Detect and track blobs in image
"""
import time
import cv2
import os
from ultralytics import YOLO


class SingleFrameDetector:
    """
    This object finds circular blobs in stereo images, and triangulates the 3D position of the correspnding landmarks
    """
    def __init__(self, model_file_path, use_cpu=False):
        """
        setup detector

        :param model_file_path: path to deep neural network model file
        :param use_cpu: force to use CPU even if GPU exists (used for timing tests)
        """
        if os.path.isfile(model_file_path):
            self.model = YOLO(model_file_path)
            if use_cpu:
                self.model.to('cpu')


    def detect(self, image, frame_resize=None):
        """
        detect objects on a specific frame

        frame can be resized to a required size (for runtime)
        in any case the actual frame for detection width and height must be dividable by 32

        :param image: (m x n x 3) image
        :param frame_resize: resize frame to (width, height) image size
                             size must be a dividable by 32!
                             None -> no resize
        """

        if frame_resize is not None:
            if len(frame_resize) != 2 or frame_resize[0] % 32 != 0 or frame_resize[1] % 32 != 0:
                raise Exception('invalid frame resize! expecting (width x height) - each dividable by 32')

        img_size = image.shape
        if len(img_size) != 3:
            raise Exception('expecting (mxnx3) color image')

        # check if frame size is dividable by 32
        frame_valid_size = image.shape[0] % 32 == 0 and image.shape[1] % 32 == 0
        if not frame_valid_size and frame_resize is None:
            frame_resize = (32 * (image.shape[1]//32), 32 * (image.shape[0]//32))

        # check if we need to resize
        no_resize = frame_resize is None or (image.shape[1] == frame_resize[0] and image.shape[0] == frame_resize[1])

        if no_resize:
            # cv2.imshow('yolo input', image)
            # cv2.waitKey(25)
            results = self.model(image)  # keep image original size
        else:
            img_resized = cv2.resize(image, frame_resize)
            results = self.model(img_resized, imgsz=frame_resize[-1::-1])  # change imgsz

        res = []
        for r in results:
            for i, box in enumerate(r.boxes):
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                confidence = box.conf[0].item()  # Confidence score
                class_id = int(box.cls[0].item())  # Class index
                # print(f"Class ID: {class_id}, Confidence: {confidence}, BBox: {xywh}")
                res.append({'id': i, 'bbox': (xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]),
                            'confidence': confidence, 'class_id': class_id})

        return res