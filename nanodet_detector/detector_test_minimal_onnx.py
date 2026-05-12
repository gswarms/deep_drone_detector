import onnxruntime
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Preprocess image
def preprocess(image_path, input_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cv2.imread failed to load image from path: {image_path}")
    orig_image = image.copy()
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # normalize to [0, 1]
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # add batch dimension

    # print(type(orig_image))
    # print(orig_image.shape)
    # print(orig_image.dtype)

    print(type(image))
    print(image.shape)
    print(image.dtype)
    return image, orig_image

# Postprocess output (example for NanoDet-like output)
def postprocess(outputs, conf_threshold=0.4):
    # This depends heavily on your NanoDet export setup.
    # Outputs should be parsed into bounding boxes, scores, and class ids.
    # Assuming [1, N, 6] format: [x1, y1, x2, y2, score, class_id]
    preds = outputs[0]  # shape: (1, N, 6)
    preds = preds.squeeze(0)

    boxes, scores, class_ids = [], [], []
    for pred in preds:
        score = pred[4]
        if score > conf_threshold:
            boxes.append(pred[:4])
            scores.append(score)
            class_ids.append(int(pred[5]))
    return boxes, scores, class_ids

# Draw boxes
def draw_boxes(image, boxes, scores, class_ids, class_names=None):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return image


if __name__ == "__main__":
    # CONFIGURATION
    ONNX_MODEL_PATH = "/home/roee/Projects/nanodet/workspace/nanodet-plus-m_320/model_best/nanodet_plus_m_320_lulav_dit_model_best.onnx"
    IMAGE_PATH = "/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/images/train/20250416_080932_1744790984435034751.jpg"
    INPUT_SIZE = (320, 320)  # use the same input size as during training

    # Load ONNX model
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run Inference
    image, orig_image = preprocess(IMAGE_PATH, INPUT_SIZE)
    outputs = session.run([output_name], {input_name: image})
    boxes, scores, class_ids = postprocess(outputs)

    # Visualize
    result_image = draw_boxes(orig_image, boxes, scores, class_ids)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Detection Result")
    plt.show()
