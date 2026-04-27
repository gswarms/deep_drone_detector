import torch
from ultralytics import YOLO
from ultralytics.utils import metrics


# --- NWD LOSS INJECTION ---
def bbox_nwd_iou(box1, box2, constant=12.8):
    """
    Normalized Gaussian Wasserstein Distance (NWD)
    Improved IoU for tiny objects.
    """
    # Extract centers and dimensions
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    cw1, ch1 = (b1_x2 - b1_x1) / 2, (b1_y2 - b1_y1) / 2
    cw2, ch2 = (b2_x2 - b2_x1) / 2, (b2_y2 - b2_y1) / 2

    # Calculate center distance and shape distance
    center_dist = (b1_x1 + cw1 - b2_x1 - cw2) ** 2 + (b1_y1 + ch1 - b2_y1 - ch2) ** 2
    shape_dist = (cw1 - cw2) ** 2 + (ch1 - ch2) ** 2

    # NWD formula
    nwd = torch.exp(-torch.sqrt(center_dist + shape_dist) / constant)
    return nwd


# Hot-patch the original bbox_iou to use NWD for small targets
_original_bbox_iou = metrics.bbox_iou


def custom_bbox_iou(box1, box2, **kwargs):
    iou = _original_bbox_iou(box1, box2, **kwargs)
    nwd = bbox_nwd_iou(box1, box2)
    # Blend IoU and NWD (0.5 weight each is standard for tiny objects)
    return 0.5 * iou + 0.5 * nwd


# Apply the patch
metrics.bbox_iou = custom_bbox_iou
# ---------------------------

if __name__ == "__main__":
    yolo_model_cfg_file = '/home/roee/Projects/datasets/interceptor_drone/uav_detection_dataset/dataset_20260330/models/yolov26n_256x256_20260330_mixed_p2_spd/yolo26n_p2_spd_256x256.yaml'
    dataset_yaml_file = '/home/roee/Projects/datasets/interceptor_drone/uav_detection_dataset/dataset_20260323/ultalytics_yolo_20260324/ultralytics_dataset_data.yaml'

    # 1. Load the architecture
    model = YOLO(yolo_model_cfg_file)

    # 2. Train with specialized settings for Pi 5 & small targets
    model.train(
        data=dataset_yaml_file,
        epochs=500,
        imgsz=256,
        batch=4,  # 256px images allow for larger batches
        optimizer='AdamW',  # Better for custom/narrow architectures
        lr0=0.001,  # Slightly lower learning rate for stability

        # --- Loss Function Tuning ---
        box=10.0,    # Increases the "penalty" for missing a box by a few pixels. For a $5 \times 5$ target, a 1-pixel error is a $20\%$ disaster.
        cls=0.5,     # Lowering classification gain prevents the model from "hallucinating" drones in background noise (False Positives).
        # iuo_t=0.25,  # The threshold for a "positive match." Keep this low so the model is rewarded even for "nearly" catching a $5 \times 5$ target.
        warmup_epochs=5.0, # Tiny targets create "noisy" gradients early on. A longer warmup prevents the model from diverging in the first hour.

        # --- Augmentation - "Atmospheric" Settings ---
        hsv_v=0.4,  # Simulates the drone being "backlit" by the sun (becoming a black silhouette) or "frontlit" (becoming a bright white spot)
        # hsv_s: 0.7,
        # noise = 0.05,  # Simulates sensor grain

        # --- Augmentation motion: ---
        # blur=0.1,    # motion blurr
        fliplr=0.5,  # always good
        flipud=0.0,  # not physical
        degrees=30.0,  # maneuvers
        translate=0.1,  # 0.4 will get the targets to the edge of the frame where convolution padding takes effect
        perspective=0.0,  # bad for small targets

        # --- Augmentation Strategy: ---
        scale=0.3,  # This is the most critical hyperparameter. It randomly zooms the image in/out by up to $90\%$.
        mosaic=1.0,  # Keep Mosaic at max for the first $80\%$ of training. It creates "synthetic" small objects by shrinking four images into one.
        close_mosaic=25,  # Disable mosaic for last 20 epochs to see "clean" 256x256 crops to refine the bounding box accuracy for the 5x5 targets.
        multi_scale=True  # This varies the input resolution (e.g., from $192$ to $320$) every few batches. It makes the kernels scale-invariant
    )

    # 3. Export for Raspberry Pi 5
    # Use half=True for 2x speedup on Pi 5 CPU via OpenVINO
    model.export(format="openvino", imgsz=256, half=True)