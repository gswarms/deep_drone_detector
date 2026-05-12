import cv2
import numpy as np
import onnxruntime as ort

# ---- CONFIG ----
# input_size = 320
# num_classes = 1
# reg_max = 7
# strides = [8, 16, 32, 64]
# score_threshold = 0.3
# nms_threshold = 0.5
# class_name = "rc-plane"  # <-- your single class
class NanoDetONNX:
    def __init__(self, onnx_path, input_size=320, score_threshold=0.52, nms_threshold=0.5, reg_max=7):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.reg_max = reg_max

        # Fixed anchor strides
        self.strides = [8, 16, 32, 64]

    def preprocess(self, image):
        h0, w0 = image.shape[:2]
        img_resized = cv2.resize(image, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_norm, (2, 0, 1))[None, ...]  # (1,3,H,W)
        return img_input, (w0, h0)

    def nms(self, boxes, scores):
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold
        )
        return indices.flatten() if len(indices) > 0 else []

    def postprocess(self, outputs, orig_w, orig_h):
        preds = outputs[0]  # (1, 2125, 33) or (2125, 33)
        if preds.ndim == 3:
            preds = preds.reshape(-1, preds.shape[-1])  # (2125, 33)

        num_anchors, num_channels = preds.shape

        cls_scores = 1.0 / (1.0 + np.exp(-preds[:, 0]))

        reg_dists = preds[:, 1:]
        num_bins = self.reg_max + 1
        reg_dists = reg_dists.reshape(num_anchors, 4, num_bins)

        mask = cls_scores > self.score_threshold
        if not np.any(mask):
            return [], []

        cls_scores = cls_scores[mask]
        reg_dists = reg_dists[mask]

        proj = np.arange(num_bins)
        # reg = np.sum(np.exp(reg_dists) / np.sum(np.exp(reg_dists), axis=-1, keepdims=True) * proj, axis=-1)
        logits = reg_dists - np.max(reg_dists, axis=-1, keepdims=True)
        prob = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        reg = np.sum(prob * proj, axis=-1)

        grids = []
        strides_arr = []
        for s in self.strides:
            feat_h = self.input_size // s
            feat_w = self.input_size // s
            yv, xv = np.meshgrid(np.arange(feat_h), np.arange(feat_w))
            grid = np.stack((xv, yv), axis=-1).reshape(-1, 2)
            grids.append(grid * s)
            strides_arr.append(np.full((feat_h * feat_w,), s))
        grids = np.concatenate(grids, axis=0)
        strides_arr = np.concatenate(strides_arr, axis=0)

        grids = grids[mask]
        stride_list = strides_arr[mask]

        x1 = grids[:, 0] - reg[:, 0] * stride_list
        y1 = grids[:, 1] - reg[:, 1] * stride_list
        x2 = grids[:, 0] + reg[:, 2] * stride_list
        y2 = grids[:, 1] + reg[:, 3] * stride_list
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        keep = self.nms(boxes, cls_scores)
        return boxes[keep], cls_scores[keep]

    def detect(self, image_path, visualize=False):
        image = cv2.imread(image_path)
        img_input, (w0, h0) = self.preprocess(image)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_input})

        boxes, scores = self.postprocess(outputs, w0, h0)

        if visualize:
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Detection", image)
            cv2.waitKey(0)

        return boxes, scores


if __name__ == "__main__":

    # ---- RUN ON REAL IMAGE ----
    image_path = "/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/images/train/20250408_124029_1744116048441849541.jpg"
    model_path = "/home/roee/Projects/nanodet/workspace/nanodet-plus-m_320/model_best/nanodet_plus_m_320_lulav_dit_model_best.onnx"

    detector = NanoDetONNX(model_path)
    boxes, scores = detector.detect(image_path, visualize=True)
    #
    # # ---- DRAW RESULTS ----
    # for box, score in zip(boxes, scores):
    #     x1, y1, x2, y2 = box.astype(int)
    #     cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(original, f"{class_name}: {score:.2f}", (x1, y1-5),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #
    # cv2.imshow("Detections", original)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
