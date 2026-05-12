import os.path

import torch
import cv2
import numpy as np
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, Logger, load_model_weight
from nanodet.data.transform import Pipeline


class NanodetDetector:
    """
    This object is a wrapper for nanodet detector including:
    - loading configuration from yaml config file
    - inference
    - post procesing NMS

    """
    def __init__(self, config_path, model_path):
        """
        init nanoder model

        :param config_path: path to configuration yaml file
        :param model_path: path to model .pth file
        """
        load_config(cfg, config_path)
        self.model = build_model(cfg.model)
        self.logger = Logger(local_rank=0)
        self.ckpt = torch.load(model_path, map_location='cpu')
        load_model_weight(self.model, self.ckpt, self.logger)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # === Preprocessing pipeline ===
        cfg.pop('keep_ratio', None)  # Avoid duplicate
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, raw_img):

        # Prepare data & meta
        input_size = tuple(cfg.data.val.input_size)  # (320, 320) for example
        meta = {
            "img": raw_img,
            "img_info": {
                "height": raw_img.shape[0],
                "width": raw_img.shape[1],
                "id": 0
            }
        }

        # Run preprocessing
        meta = self.pipeline(meta, meta, input_size)

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
            results = self.model.inference(meta)  # ✅ Correct format: list of dicts

        # NMS
        nms_results = nms_nanodet(results, iou_threshold=0.5)

        return nms_results


def nms(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes (np.ndarray): Bounding boxes (N, 4) format: [x1, y1, x2, y2]
        scores (np.ndarray): Confidence scores (N,)
        iou_threshold (float): IoU threshold for suppression

    Returns:
        keep (list): Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(float)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]  # Sort boxes by score (desc)
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of the kept box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU below the threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def nms_nanodet(results, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes, for nanodet results.

    :param results: nanodet results
    :param iou_threshold: [0,1]
    :return: nanodet results
    """
    result = results[0]

    nms_results = {}
    for class_id, detections in result.items():
        bboxes = np.array([det[:4] for det in detections])
        scores = np.array([det[4] for det in detections])
        idx = nms(bboxes, scores, iou_threshold=iou_threshold)
        bboxes_nms = bboxes[idx, :]
        scores_nms = scores[idx].reshape((-1,1))
        nms_results[class_id] = list(np.hstack((bboxes_nms, scores_nms)) )

    return nms_results


if __name__ == "__main__":
    # Config and model loading
    config_path = "/home/roee/Projects/nanodet/config/nanodet-plus-m_320_lulav_dit.yml"
    model_path = "/home/roee/Projects/nanodet/workspace/nanodet-plus-m_320/model_best/model_best.ckpt"
    image_path = "/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/images/train/20250416_080932_1744790984435034751.jpg"

    # init model
    nnd_detector = NanodetDetector(config_path, model_path)

    # load image
    if not os.path.isfile(image_path):
        raise Exception('image file: {} not found!'.format(image_path))
    raw_img = cv2.imread(image_path)
    assert raw_img is not None, f"Image not found: {image_path}"

    # infer
    results = nnd_detector.inference(raw_img)

    # Draw detections
    for class_id, detections in results.items():
        for det in detections:
            x1, y1, x2, y2, score = det[:5]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_id}: {score:.2f}"
            cv2.putText(raw_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Result", raw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
