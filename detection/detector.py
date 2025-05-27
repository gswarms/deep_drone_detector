""" Detect and track blobs in image
"""
import time
import cv2
import os
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort


class SingleFrameDetector:
    """
    This object finds circular blobs in stereo images, and triangulates the 3D position of the correspnding landmarks
    """
    def __init__(self, model_file_path, use_cpu=False, verbose=False):
        """
        setup detection

        :param model_file_path: path to deep neural network model file
        :param use_cpu: force to use CPU even if GPU exists (used for timing tests)
        """
        self.verbose = verbose
        self.model_type = None
        self.onnx_input_name = None
        self.onnx_input_size = None
        if os.path.isfile(model_file_path):
            model_file_name, model_file_extension = os.path.splitext(model_file_path)

            if model_file_extension == '.pt':
                self.model = YOLO(model_file_path)
                self.model_type = 'pt'
                if use_cpu:
                    self.model.to('cpu')
                    if self.verbose:
                        print('using CPU only!')
                if self.verbose:
                    print('using pt model:')
                    self.model.info(verbose=True)

            elif model_file_extension == '.onnx':
                # Load ONNX model
                self.model = ort.InferenceSession(model_file_path)
                self.model_type = 'onnx'
                input_info = self.model.get_inputs()[0]
                self.onnx_input_name = input_info.name
                self.onnx_input_size = input_info.shape[2:]
                if self.verbose:
                    print('using onnx model:')
                    print(input_info)

            else:
                raise Exception('invalid model file extention: {}!'.format(model_file_extension))



    def detect(self, image, frame_resize=None):
        """
        detect objects on a specific frame

        frame can be resized to a required size (for runtime)
        in any case the actual frame for detection width and height must be dividable by 32

        :param image: (m x n x 3) image
        :param frame_resize: resize frame to (width, height) image size
                             size must be a dividable by 32!
                             None -> no resize
                             This is only valid for pt models!
                             onnx models always resize to onnx predefined shape.
        """

        img_size = image.shape
        if len(img_size) != 3:
            raise Exception('expecting (mxnx3) color image')

        if self.model_type == 'pt':

            # check frame resize input
            if frame_resize is not None:
                if len(frame_resize) != 2 or frame_resize[0] % 32 != 0 or frame_resize[1] % 32 != 0:
                    raise Exception('invalid frame resize! expecting (width x height) - each dividable by 32')
            # check if frame size is dividable by 32
            frame_valid_size = image.shape[0] % 32 == 0 and image.shape[1] % 32 == 0
            if not frame_valid_size and frame_resize is None:
                frame_resize = (32 * (image.shape[1] // 32), 32 * (image.shape[0] // 32))

            # check if image resize is needed
            if frame_resize is None:
                need_resize = False
            else:
                need_resize = image.shape[1] != frame_resize[0] or image.shape[0] != frame_resize[1]

            # model predict
            if need_resize:
                img_resized = cv2.resize(image, frame_resize)
                outputs = self.model(img_resized, imgsz=(img_resized.shape[:2]), stream=False, verbose=self.verbose)
            else:
                outputs = self.model(image, imgsz=(image.shape[:2]), stream=False, verbose=self.verbose)
            results = self._pt_postprocess(outputs)

        elif self.model_type == 'onnx':
            # onnx preprocessing
            image_input = self._onnx_preprocess(image)
            # onnx inference
            outputs = self.model.run(None, {self.onnx_input_name: image_input})
            # onnx post processing
            results = self._onnx_postprocess(outputs, conf_threshold=0.4, iou_threshold=0.5)

        else:
            raise Exception('invalid model type: {}!'.format(self.model_type))


        return results

    def _model_predict(self, image, input_shape=(640, 480)):

        need_resize = image.shape[1] != input_shape[0] or image.shape[0] != input_shape[1]

        if self.model_type == 'pt':
            if need_resize:
                img_resized = cv2.resize(image, input_shape)
                outputs = self.model(img_resized, imgsz=(img_resized.shape[:2]), stream=False, verbose=self.verbose)
            else:
                outputs = self.model(image, imgsz=(image.shape[:2]), stream=False, verbose=self.verbose)
            results = self._pt_postprocess(outputs)

        elif self.model_type == 'onnx':
            # onnx preprocessing
            image_input = self._onnx_preprocess(image)
            # onnx inference
            outputs = self.model.run(None, {self.onnx_input_name: image_input})
            # onnx post processing
            results = self._onnx_postprocess(outputs, conf_threshold=0.4, iou_threshold=0.5)

        else:
            raise Exception('invalid model type: {}!'.format(self.model_type))

        return results

    def _onnx_preprocess(self, image_bgr):
        """
        pre-process image for onnx.
        onnx must have image in the expected size, rgb, normalized to [0,1] and reshaped to HWC → CHW.

        :param image_bgr: [mxnx3] bgr image
        :return: image_input: [3xmxn] image ready for onnx
        """
        # RGB -> BGR
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # resize image to onnx expected size
        need_resize = image_rgb.shape[1] != self.onnx_input_size[0] or image_rgb.shape[0] != self.onnx_input_size[1]
        if need_resize:
            image_resized = cv2.resize(image_rgb, self.onnx_input_size)
        else:
            image_resized = image_rgb

        # normalize image values to [0,1]
        image_normalized = image_resized / 255.0  # Normalize to 0-1

        # reshape HWC → CHW
        image_transposed = np.transpose(image_normalized, (2, 0, 1))  # HWC to CHW
        image_input = np.expand_dims(image_transposed, axis=0).astype(np.float32)
        return image_input

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _onnx_postprocess(self, predictions, conf_threshold=0.4, iou_threshold=0.5):
        """
        post-processing onnx results:
        1. Translate objectness score and class score to confidence level
        2. Converts from YOLO center-based box format to corner coordinates
        3. Non-Maximal Suppression - Suppresses overlapping boxes for cleaner results

        *** important note:
            it matters if the onnx is exported with 'simplify'=True or 'simplify'=False!
            if 'simplify'=True the results are (m,5,N) where:
                - m = batch size (number of images)
                - data per detection is: [x, y, w, h, confidence]
                - N = number of detections

            if 'simplify'=False the results are (m,5+num_classes,N)
                - m = batch size (number of images)
                - data per detection is: [x, y, w, h, objectness_confidence, class1_confidence, ... ,classn_confidence]
                - N = number of detections

            so different format is needed in post process.


        :param predictions:
        :param conf_threshold:
        :param iou_threshold:
        :return:
        """
        boxes = []
        confidences = []
        class_ids = []

        # predictions: [1, N, 84] → Remove batch dimension
        preds = predictions[0]

        for pred in preds:

            if pred.shape[0] == 5:  # simplified

                confidences_tmp = pred[4, :]
                idx = confidences_tmp > conf_threshold

                x, y, w, h = pred[0:4, idx]
                confidences = pred[4, idx]

                # Convert to x1, y1, x2, y2
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                boxes = np.vstack((x1, y1, x2, y2)).T
                class_ids = np.zeros(boxes.shape[0])


            elif pred.shape[0] > 5:  # non-simplified

                x, y, w, h = pred[0:4]
                obj_conf = self._sigmoid(pred[4])  # objectness score

                class_scores = pred[5:]
                class_scores = self._sigmoid(class_scores)  # sigmoid for class confidence
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                final_conf = obj_conf * class_conf
                if final_conf > conf_threshold:
                    # Convert to x1, y1, x2, y2
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(final_conf))
                    class_ids.append(class_id)

                else:
                    raise Exception('invalod onnx output size!')

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
        results = []
        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            # convert back to [xtl,ytl,w,h]
            results.append({'bbox': [boxes[i][0], boxes[i][1], boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]],
                            'confidence': confidences[i], 'class_id': class_ids[i]})
        return results


    def _pt_postprocess(self, ultralytics_results):
        """
        just convert ultralytics pt model results to common format
        1. translate to simple dicts
        2. Converts from YOLO center-based box format to corner coordinates

        *** mns already performed by the pt model!

        :param ultralytics_results:
        :return:
        """
        results = []
        for r in ultralytics_results:
            for i, box in enumerate(r.boxes):
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                confidence = box.conf[0].item()  # Confidence score
                class_id = int(box.cls[0].item())  # Class index
                # print(f"Class ID: {class_id}, Confidence: {confidence}, BBox: {xywh}")
                results.append({'bbox': (xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]),
                            'confidence': confidence, 'class_id': class_id})
        return results
