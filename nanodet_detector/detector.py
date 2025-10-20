""" Detect and track blobs in image
"""
import os
import cv2
import numpy as np
import torch
# import torch.nn.functional as F
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, Logger, load_model_weight
from nanodet.data.transform import Pipeline
import onnxruntime as ort

# TODO: single frame nanodet seems to work! add onnx functionality, add CPU/GPU options

class NanodetDetector:
    """
    This object is a wrapper for nanodet detector including:
    - loading configuration from yaml config file
    - inference
    - post procesing NMS
    """
    def __init__(self, model_file_path, config_path=None, use_cpu=False, verbose=False):
        """
        init nanoder model

        :param config_path: path to configuration yaml file
        :param model_file_path: path to deep neural network model file
                        We support two types of neural network models:
                        - .pt - better for development or in any case we have GPGPU
                        - .onnx - better for CPU based platforms
                        * Note: Exporting a model to .onnx can be done in various ways.\
                                we currently support "simplified=True", and "uint8=False"
                        * Note: In onnx model, the image will automatically be resized to .onnx predefined image size!
        :param use_cpu: force to use CPU even if GPU exists (used for timing tests)
        """

        self.verbose = verbose
        if not os.path.isfile(model_file_path):
            raise Exception('model file: {} not found!'.format(model_file_path))

        model_file_name, model_file_extension = os.path.splitext(model_file_path)

        if model_file_extension in ['.pt', '.pth', '.ckpt']:
            self.model_type = 'pt'
            if self.verbose:
                print('using {} model'.format(self.model_type))
            if not os.path.isfile(config_path):
                raise Exception('config file: {} not found!'.format(config_path))

            # Load pytorch model
            load_config(cfg, config_path)
            self.model = build_model(cfg.model)
            self.logger = Logger(local_rank=0)
            self.ckpt = torch.load(model_file_path, map_location='cpu')
            load_model_weight(self.model, self.ckpt, self.logger)
            self.model.eval()

            if use_cpu:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            if self.verbose:
                print('torch device: {}'.format(self.device.type))

            # === Preprocessing pipeline ===
            cfg.pop('keep_ratio', None)  # Avoid duplicate
            self.input_size = tuple(cfg.data.val.input_size)  # (320, 320) for example

            self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
            self.verbose = verbose
            self.onnx_input_name = None
            self.img_resized = np.empty((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)

        elif model_file_extension == '.onnx':
            self.model_type = 'onnx'
            if self.verbose:
                print('using {}} model'.format(self.model_type))

            load_config(cfg, config_path)
            self.num_classes = cfg.model.arch.head.num_classes

            # Load ONNX model
            self.model = ort.InferenceSession(model_file_path)
            input_info = self.model.get_inputs()[0]
            if self.verbose:
                print(input_info)
            self.onnx_input_name = input_info.name
            self.input_size = input_info.shape[2:]

            H, W = self.input_size[1], self.input_size[0]
            self.img_resized = np.empty((H, W, 3), dtype=np.uint8)
            self.onnx_img_uint_rgb = np.empty((H, W, 3), dtype=np.uint8)
            self.onnx_img_normlized = np.empty((H, W, 3), dtype=np.float32)
            self.onnx_img = np.empty((1, 3, H, W), dtype=np.float32)


    def detect(self, image, frame_resize=False, conf_threshold=0.4, nms_iou_threshold=0.5, max_num_detections=20):
        """
        detection step

        :param image: 3 channel color image
        :param frame_resize: if true: the image will be resized automatically if it does not fit the model input size.
                              if false: the imagesize will be checked, and raise error if it doesn't fit the model input size.
        :param conf_threshold: confidence thresholds
        :param nms_iou_threshold: iou threshold for nms
        :param max_num_detections: max number of detections
        :return:
        """

        if frame_resize is None:  # comply with general detector interface
            frame_resize = False

        img_size = image.shape
        if len(img_size) != 3:
            raise Exception('expecting (mxnx3) color image')

        img_resized = False
        resize_scale_x = None
        resize_scale_y = None
        if img_size[0] != self.input_size[0] or img_size[1] != self.input_size[1]:
            if frame_resize:
                self.img_resized[:] = cv2.resize(image, (self.input_size[0], self.input_size[1], 3))
                img_resized = True
                resize_scale_x = image.shape[1] / self.img_resized.shape[1]
                resize_scale_y = image.shape[0] / self.img_resized.shape[0]
            else:
                raise Exception('invalid image size! image size is: {} does not fit model input size: {}'.format(img_size, self.input_size))
        else:
            self.img_resized = image

        if self.model_type == 'pt':
            # Prepare data & meta
            meta = {
                "img": self.img_resized,
                "img_info": {
                    "height": self.img_resized.shape[0],
                    "width": self.img_resized.shape[1],
                    "id": 0
                }
            }

            # preprocessing
            meta = self.pipeline(meta, meta, self.input_size)

            # Move tensor to device
            img = meta["img"]  # numpy array, shape: (C, H, W) or (H, W, C)
            # If shape is (H, W, C), transpose to (C, H, W)
            if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
                img = img.transpose(2, 0, 1)
            # img = np.expand_dims(img, axis=0)
            img_tensor = torch.from_numpy(img).float()
            meta["img"] = img_tensor.unsqueeze(0).to(self.device)  # Add batch dim and move to device

            # nanodet needs inputs to be in multiple images tensor/list format
            meta["img_info"]["height"] = torch.from_numpy(np.array(meta["img_info"]["height"])).int().unsqueeze(0)
            meta["img_info"]["width"] = torch.from_numpy(np.array(meta["img_info"]["width"])).int().unsqueeze(0)
            meta["img_info"]["id"] = torch.from_numpy(np.array(meta["img_info"]["id"])).int().unsqueeze(0)
            meta['warp_matrix'] = [meta['warp_matrix']]

            # Run inference
            with torch.no_grad():
                nanodet_results = self.model.inference(meta)  # ✅ Correct format: list of dicts

            nms_results = self._pt_postprocess(nanodet_results, conf_threshold=conf_threshold,
                                               nms_iou_threshold=nms_iou_threshold,
                                               max_num_detections=max_num_detections)
            # NMS
            # nms_results = nms_nanodet(results, iou_threshold=nms_iou_threshold)

        elif self.model_type == 'onnx':
            # onnx preprocessing
            self._onnx_preprocess(self.img_resized)
            # onnx inference
            outputs = self.model.run(None, {self.onnx_input_name: self.onnx_img})
            # onnx post processing
            # nms_results = self._onnx_postprocess(outputs, conf_threshold=conf_threshold, nms_iou_threshold=nms_iou_threshold,
            #                                  max_num_detections=max_num_detections)

            nms_results = self._onnx_postprocess(outputs, input_size = (320, 320), strides = [8, 16, 32, 64],
                            num_classes = 1, reg_max = 7,
                            score_threshold = 0.35, nms_threshold = 0.5)

        else:
            raise Exception('invalid model type: {}!'.format(self.model_type))

        if img_resized:
            raise Exception('converting to original frame coordinates not supported yet!')
            # convert result bbox to original image coordinates
            # resize_scale_x = ???
            # resize_scale_y = ???

        return nms_results


    def _onnx_preprocess(self, image):
        """
        pre-process image for onnx.
        onnx must have image in the expected size, rgb, normalized to [0,1] and reshaped to HWC → CHW.

        * Note: image will automatically be resized to .onnx predefined image size!

        :param image_bgr: [mxnx3] bgr image
        :return: image_input: [3xmxn] image ready for onnx
        """
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # orig_h, orig_w = img.shape[:2]
        #
        # # Resize and normalize
        # resized = cv2.resize(img, input_size)
        # resized = resized.astype(np.float32) / 255.0
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # resized = (resized - mean) / std
        #
        # # HWC to CHW
        # input_tensor = resized.transpose(2, 0, 1)
        # input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 3, H, W)
        # return input_tensor, (orig_w, orig_h)



        # check image size fits onnx expected size
        if image.shape[1] != self.input_size[0] or image.shape[0] != self.input_size[1]:
            raise Exception('image size {} does not fit model expected size: {}'.format(image.shape, self.input_size))
        if image.shape[2] != 3:
            raise Exception('image size {} - expecting a BGR image: (MxNx3)!'.format(image.shape))

        # RGB -> BGR
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=self.onnx_img_uint_rgb)

        # normalize image values to [0,1]
        np.copyto(self.onnx_img_normlized, self.onnx_img_uint_rgb)
        self.onnx_img_normlized /= 255.0  # in-place normalization

        # Normalize
        # mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        # std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        # img_normalized = (img_float - mean) / std

        # reshape HWC → CHW
        self.onnx_img[0][:] = np.transpose(self.onnx_img_normlized, (2, 0, 1))
        # np.copyto(self.onnx_img[0], np.transpose(self.onnx_img_normlized, (2, 0, 1)))

        # img_chw = np.transpose(img_normalized, (2, 0, 1))
        # img_input = np.expand_dims(img_chw, axis=0)  # Add batch dim

        return

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _onnx_postprocess(self, predictions, input_size=(320,320), strides=[8,16,32, 64],
                num_classes=1, reg_max=8,
                score_threshold=0.35, nms_threshold=0.5):
        """
        post-processing onnx results:
        1. Translate objectness score and class score to confidence level
        2. Converts from YOLO center-based box format to corner coordinates
        3. Non-Maximal Suppression - Suppresses overlapping boxes for cleaner results

        :param predictions: nanodet-plus-m onnx model output, shape [(1, N, num_classes + 4 * reg_max)]
        :param conf_threshold: detections with confidence scores lower than this threshold are discarded.
        :param nms_iou_threshold: The threshold used in NMS to decide whether boxes overlap too much with respect to
                                  Intersection over Union (IoU). If the IoU is greater than this value,
                                  the box with the lower score is suppressed.
        :param max_num_detections: Keeps at most k detections. Useful when you only want the top results.
        :param strides: list of strides used in the model ([8, 16, 32] is common for nanodet-plus-m onnx)
        :return:
        """

        assert predictions[0].ndim == 3 and predictions[0].shape[0] == 1
        pred = predictions[0][0]  # shape (N, C + 4*reg_max)
        N, C4 = pred.shape
        cls_logits = pred[:, :num_classes]  # (N, num_classes)
        bbox_dist = pred[:, num_classes:]  # (N, 4*reg_max)

        # for single class scenario, get score directly
        # if multiple classes, you would take max over axis=1
        if num_classes == 1:
            cls_scores = cls_logits[:, 0]
            class_ids = np.zeros_like(cls_scores, dtype=np.int32)
        else:
            class_ids = np.argmax(cls_logits, axis=1)
            cls_scores = cls_logits[np.arange(N), class_ids]

        # filter by score threshold
        keep = cls_scores > score_threshold
        if not np.any(keep):
            return []

        cls_scores = cls_scores[keep]
        class_ids = class_ids[keep]
        bbox_dist = bbox_dist[keep]

        # generate centers
        center_points, stride_for_pt = generate_center_points(input_size, strides)  # (total_pts, 2)
        center_points = center_points[keep]
        stride_for_pt = stride_for_pt[keep]  # Match filtered predictions

        # decode boxes
        boxes = decode_boxes(bbox_dist, center_points, stride_for_pt, reg_max=reg_max, input_size=input_size )


        # optional: clip boxes to image bounds (based on input size)
        # but note these are still relative to input image scale

        # NMS
        # OpenCV’s NMS wants boxes in list of [x,y,w,h], so convert
        boxes_xywh = []
        for b in boxes:
            x1, y1, x2, y2 = b
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        boxes_xywh = np.array(boxes_xywh, dtype=np.float32)

        # cv2.dnn.NMSBoxes requires list of lists and list of scores
        boxes_list = boxes_xywh.tolist()
        scores_list = cls_scores.tolist()

        nms_indices = cv2.dnn.NMSBoxes(boxes_list, scores_list,
                                       score_threshold, nms_threshold)
        detections = []
        if len(nms_indices) > 0:
            # nms_indices is a list of [i]
            for idx in nms_indices:
                i = idx[0] if isinstance(idx, (tuple, list, np.ndarray)) else idx
                x, y, w, h = boxes_xywh[i]
                detections.append({'bbox': [x, y, x + w, y + h],
                     'confidence': cls_scores[i], 'class_id': int(class_ids[i])})
                #
                #     [
                #     x, y, x + w, y + h,
                #     cls_scores[i], int(class_ids[i])
                # ])

        return detections

    @staticmethod
    def _pt_postprocess(nanodet_results, conf_threshold=0.4, nms_iou_threshold=0.5, max_num_detections=10):
        """
        Decode ultralytics pt model results to common format:
        1. Translate to simple dicts
        2. Converts from YOLO center-based box format to corner coordinates

        Apply BBOX based Non-Maximal Supression.
        * Note: this is already done by uptralytics model. But we noticed overlapping bbox yolo_detector sometimes
                Therefore we apply MNS again, and it solves the problem.

        *** mns already performed by the pt model!

        :param nanodet_results:
        :param conf_threshold: detections with confidence scores lower than this threshold are discarded.
        :param nms_iou_threshold: The threshold used in NMS to decide whether boxes overlap too much with respect to
                                  Intersection over Union (IoU). If the IoU is greater than this value,
                                  the box with the lower score is suppressed.
        :param max_num_detections: Keeps at most k detections. Useful when you only want the top results.
        :return:
        """

        result = nanodet_results[0]  # nanodet returns results for a batch
                                     # in this class we infer a single image
        boxes = []
        confidences = []
        class_ids = []
        for class_id, detections in result.items():
            for det in detections:
                boxes.append(det[:4])  # (xtl, ytl, w, h))
                confidences.append(det[4])
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_iou_threshold, top_k=max_num_detections)
        nms_results = []
        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            nms_results.append({'bbox': [boxes[i][0], boxes[i][1], boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]],
                            'confidence': confidences[i], 'class_id': class_ids[i]})

        return nms_results


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def stable_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)  # subtract max for numerical stability
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# def generate_center_points(input_size=(320,320), strides=[8,16,32]):
#     """
#     Generate anchor center points (cx, cy) for each stride and feature map.
#     Returns (num_points, 2) array.
#     """
#     points = []
#     for stride in strides:
#         feat_w = input_size[0] // stride
#         feat_h = input_size[1] // stride
#         for y in range(feat_h):
#             for x in range(feat_w):
#                 cx = (x + 0.5) * stride
#                 cy = (y + 0.5) * stride
#                 points.append([cx, cy])
#     return np.array(points, dtype=np.float32)
def generate_center_points(input_size=(320, 320), strides=[8, 16, 32, 64]):
    center_points = []
    stride_for_pt = []
    for stride in strides:  # ascending order
        feat_w = int(np.ceil(input_size[0] / stride))
        feat_h = int(np.ceil(input_size[1] / stride))
        for y in range(feat_h):
            for x in range(feat_w):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                center_points.append([cx, cy])
                stride_for_pt.append(stride)  # must match order
    return np.array(center_points, dtype=np.float32), np.array(stride_for_pt, dtype=np.float32)

def decode_boxes(distribution_preds, center_points, strides, reg_max=8, input_size=[320,320]):
    """
    distribution_preds: shape (N, 4*reg_max)
    center_points: (N, 2)
    strides: list of strides in the same tiling order as center_points were generated
    Returns: boxes (N, 4) in format x1,y1,x2,y2
    """
    # distribution_preds reshape to (N, 4, reg_max)
    N = distribution_preds.shape[0]
    dist = distribution_preds.reshape(N, 4, reg_max + 1)
    # apply softmax on last dim
    # dist_prob = softmax(dist, axis=2)  # shape (N,4,reg_max)
    dist_prob = stable_softmax(dist, axis=2)  # shape (N,4,reg_max)
    #
    # left_logits = dist[:, 0, :]
    # top_logits = dist[:, 1, :]
    # right_logits = dist[:, 2, :]
    # bottom_logits = dist[:, 3, :]
    # dist_prob_left = stable_softmax(left_logits, axis=1)  # shape (N,4,reg_max)
    # dist_prob_top = stable_softmax(top_logits, axis=1)  # shape (N,4,reg_max)
    # dist_prob_right = stable_softmax(right_logits, axis=1)  # shape (N,4,reg_max)
    # dist_prob_bottom = stable_softmax(bottom_logits, axis=1)  # shape (N,4,reg_max)



    # project: [0,1,2...reg_max-1]
    proj = np.arange(reg_max + 1, dtype=np.float32)
    # expectation: sum(p * i) for each side
    # gives (N, 4)
    distances = np.sum(dist_prob * proj, axis=2)
    # Now scale by stride
    # But note: center_points includes mixed strides; we need a stride array per point
    # One simple way: build a parallel list of stride per point when generating center_points
    # For simplicity, assume same order: first stride[0] block, then stride[1], etc.
    # We'll build a stride_for_pt array:

    print("Distances (before stride scale):", distances[:5])
    print("Strides for pts:", strides[:5])
    print("Distances (after stride scale):", distances[:5] * strides[:5, None])

    # distances_left = np.sum(dist_prob_left * proj, axis=1)
    # distances_top = np.sum(dist_prob_top * proj, axis=1)
    # distances_right = np.sum(dist_prob_right * proj, axis=1)
    # distances_bottom = np.sum(dist_prob_bottom * proj, axis=1)


    # Expand to (N,4)
    stride_mat = np.expand_dims(strides, axis=1).repeat(4, axis=1)
    # distances in pixel units
    distances = distances * stride_mat  # shape (N,4)
    # compute boxes
    x1 = center_points[:, 0] - distances[:, 0]
    y1 = center_points[:, 1] - distances[:, 1]
    x2 = center_points[:, 0] + distances[:, 2]
    y2 = center_points[:, 1] + distances[:, 3]

    boxes = np.stack([x1, y1, x2, y2], axis=-1)

    print("Center point:", center_points)
    print("Stride:", strides)
    print("Distances (px):", distances)
    print("Decoded box:", boxes)

    return boxes

