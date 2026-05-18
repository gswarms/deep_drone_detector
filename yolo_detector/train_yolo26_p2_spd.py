import os
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


def train_yolo26_p2spd_v0(dataset_yaml_file, yolo_model_cfg_file):

    # 1. Load the architecture
    model = YOLO(yolo_model_cfg_file)

    # 3. Train with specialized settings for Pi 5 & small targets
    model.train(
        data=dataset_yaml_file,
        epochs=400,
        imgsz=256,
        batch=-1,  # 256px images allow for larger batches
        optimizer='AdamW',  # Better for custom/narrow architectures
        lr0=0.001,  # Slightly lower learning rate for stability

        # --- Loss Function Tuning ---
        dfl=1.5,     # The "penalty" for correct localization (edge positions)
        box=7.5,    # The "penalty" for bbox accuracy in total LOSS
                     # increase to put more weight on correct bbox
        cls=2.0,     # The "penalty" for correct classification in the total loss
                     # lower to prevent the model from "hallucinating" drones in background noise.
                     # increase to detect more allowing also false positive.
        # iuo_t=0.25,  # The threshold for a "positive match." Keep this low so the model is rewarded even for "nearly" catching a $5 \times 5$ target.
        warmup_epochs=5.0, # Tiny targets create "noisy" gradients early on. A longer warmup prevents the model from diverging in the first hour.

        # --- Augmentation - "Atmospheric" Settings ---
        augment=True,
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
        copy_paste=0.3,  # aggressive because of 50% background images
        multi_scale=False,  # This varies the input resolution (e.g., from $192$ to $320$) every few batches. It makes the kernels scale-invariant

        device = 0,  # Explicitly forces GPU 0
        project = "uav_detection",
        name = "p2spd_baseline_training_cfg"
    )

    final_weights = os.path.join("runs", "detect", "uav_detection", "p2spd_baseline_training_cfg", "weights", "best.pt")
    return final_weights


def train_yolo26_from_scratch(dataset_yaml_file, yolo_model_cfg_file):
    # =========================================================================
    # TRAINING: train from scratch
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Training Custom Head with Frozen Backbone Layers 0-6")
    print("=" * 60)

    # 1. Load the architecture
    model = YOLO(yolo_model_cfg_file)

    # 2. Train with specialized settings for Pi 5 & small targets
    model.train(
        data=dataset_yaml_file,
        epochs=400,
        imgsz=256,
        batch=-1,  # 256px images allow for larger batches
        optimizer='AdamW',  # Better for custom/narrow architectures
        lr0=0.001,  # Slightly lower learning rate for stability
        cos_lr=True,  # Smooths out final convergence for scratch training

        # --- Loss Function Tuning ---
        dfl=1.5,     # The "penalty" for correct localization (edge positions)
        box=7.5,    # The "penalty" for bbox accuracy in total LOSS
                     # increase to put more weight on correct bbox
        cls=1.0,     # The "penalty" for correct classification in the total loss
                     # lower to prevent the model from "hallucinating" drones in background noise.
                     # increase to detect more allowing also false positive.
        # iuo_t=0.25,  # The threshold for a "positive match." Keep this low so the model is rewarded even for "nearly" catching a $5 \times 5$ target.
        warmup_epochs=5.0, # Tiny targets create "noisy" gradients early on. A longer warmup prevents the model from diverging in the first hour.

        # --- Augmentation - "Atmospheric" Settings ---
        augment=True,
        hsv_v=0.4,  # Simulates the drone being "backlit" by the sun (becoming a black silhouette) or "frontlit" (becoming a bright white spot)
        # hsv_s: 0.7,
        # noise = 0.05,  # Simulates sensor grain
        erasing=0.0,  # CRITICAL: Prevents patches from deleting 5x5 targets

        # --- Augmentation motion: ---
        # blur=0.1,    # motion blurr
        fliplr=0.5,  # always good
        flipud=0.0,  # not physical
        degrees=30.0,  # maneuvers
        translate=0.1,  # 0.4 will get the targets to the edge of the frame where convolution padding takes effect
        perspective=0.0,  # bad for small targets

        # --- Augmentation Strategy: ---
        scale=0.3,  # This is the most critical hyperparameter. It randomly zooms the image in/out by up to $90\%$.
                    # might be invalid doe detection (only good for segmentation)
        mosaic=1.0,  # Keep Mosaic at max for the first $80\%$ of training. It creates "synthetic" small objects by shrinking four images into one.
        close_mosaic=25,  # Disable mosaic for last 20 epochs to see "clean" 256x256 crops to refine the bounding box accuracy for the 5x5 targets.
        copy_paste=0.3,  # aggressive because of 50% background images
        multi_scale=False,  # This varies the input resolution (e.g., from $192$ to $320$) every few batches. It makes the kernels scale-invariant

        device = 0,  # Explicitly forces GPU 0
        project = "uav_detection",
        name = "training_from_scratch"
    )
    final_weights = os.path.join("runs", "detect", "uav_detection", "training_from_scratch", "weights", "best.pt")
    return final_weights

def train_yolo26_from_prior_weights(dataset_yaml_file, yolo_model_cfg_file, pretrained_checkpoint):
    # =========================================================================
    # TRAINING STAGE 1: Warm up the untrained P2 Detection Head
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Training Custom Head with Frozen Backbone Layers 0-6")
    print("=" * 60)

    # 1. Initialize the network architecture from your custom YAML layout
    model_stage1 = YOLO(yolo_model_cfg_file)

    # 2. Inject pretrained weights via shape-matching (resets the custom layers)
    model_stage1.load(pretrained_checkpoint)

    # 3. Train for a short burst to adapt the blind head to the coordinate grid
    model_stage1.train(
        data=dataset_yaml_file,
        epochs=25,
        imgsz=256,
        batch=-1,  # 256px images allow for larger batches
        optimizer='AdamW',  # Better for custom/narrow architectures
        lr0=0.01,  # Slightly lower learning rate for stability
        freeze=7,  # Freezes layers 0-6 (Focus layer + C3k2 blocks)
        cos_lr=True,  # Smooth learning rate decay

        # --- Loss Function Tuning ---
        box=7.5,     # The "penalty" for bbox accuracy in total LOSS
                     # increase to put more weight on correct bbox
        cls=1.0,     # The "penalty" for correct classification in the total loss
                     # lower to prevent the model from "hallucinating" drones in background noise.
                     # increase to detect more allowing also false positive.
        # iuo_t=0.25,  # The threshold for a "positive match." Keep this low so the model is rewarded even for "nearly" catching a $5 \times 5$ target.
        warmup_epochs=3.0, # Tiny targets create "noisy" gradients early on. A longer warmup prevents the model from diverging in the first hour.

        # --- Augmentation - "Atmospheric" Settings ---
        augment=True,
        hsv_v=0.4,  # Simulates the drone being "backlit" by the sun (becoming a black silhouette) or "frontlit" (becoming a bright white spot)
        # hsv_s: 0.7,
        # noise = 0.05,  # Simulates sensor grain
        erasing=0.0,  # CRITICAL: Prevents deleting tiny 5x5 drone pixels

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
        copy_paste=0.3,  # aggressive because of 50% background images
        multi_scale=False,  # This varies the input resolution (e.g., from $192$ to $320$) every few batches. It makes the kernels scale-invariant

        device = 0,  # Explicitly forces GPU 0
        project = "uav_detection",
        name = "stage1_head_warmup_v2"
    )

    # =========================================================================
    # TRAINING STAGE 2: Full Fine-Tuning (Unfrozen Backbone)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-Tuning Full Architecture Together")
    print("=" * 60)

    # 1. Locate the best model generated from Stage 1
    stage1_weights = os.path.join("runs", "detect", "uav_detection", "stage1_head_warmup_v2", "weights", "best.pt")
    if not os.path.exists(stage1_weights):
        raise FileNotFoundError(f"Could not find Stage 1 weights at {stage1_weights}")
    model_stage2 = YOLO(stage1_weights)

    # 2. Open up all weights for fine-tuning at a lower learning rate
    model_stage2.train(
        data=dataset_yaml_file,
        epochs=400,
        imgsz=256,
        batch=-1,  # 256px images allow for larger batches
        optimizer='AdamW',  # Better for custom/narrow architectures
        lr0=0.001,  # Slightly lower learning rate for stability
        cos_lr=True,  # Smooths out final convergence for scratch training

        # --- Loss Function Tuning ---
        box=7.5,    # The "penalty" for bbox accuracy in total LOSS
                     # increase to put more weight on correct bbox
        cls=1.0,     # The "penalty" for correct classification in the total loss
                     # lower to prevent the model from "hallucinating" drones in background noise.
                     # increase to detect more allowing also false positive.
        # iuo_t=0.25,  # The threshold for a "positive match." Keep this low so the model is rewarded even for "nearly" catching a $5 \times 5$ target.
        warmup_epochs=5.0, # Tiny targets create "noisy" gradients early on. A longer warmup prevents the model from diverging in the first hour.

        # --- Augmentation - "Atmospheric" Settings ---
        augment=True,
        hsv_v=0.4,  # Simulates the drone being "backlit" by the sun (becoming a black silhouette) or "frontlit" (becoming a bright white spot)
        # hsv_s: 0.7,
        # noise = 0.05,  # Simulates sensor grain
        erasing=0.0,  # CRITICAL: Prevents patches from deleting 5x5 targets

        # --- Augmentation motion: ---
        # blur=0.1,    # motion blurr
        fliplr=0.5,  # always good
        flipud=0.0,  # not physical
        degrees=30.0,  # maneuvers
        translate=0.1,  # 0.4 will get the targets to the edge of the frame where convolution padding takes effect
        perspective=0.0,  # bad for small targets

        # --- Augmentation Strategy: ---
        scale=0.3,  # This is the most critical hyperparameter. It randomly zooms the image in/out by up to $90\%$.
                    # might be invalid doe detection (only good for segmentation)
        mosaic=1.0,  # Keep Mosaic at max for the first $80\%$ of training. It creates "synthetic" small objects by shrinking four images into one.
        close_mosaic=25,  # Disable mosaic for last 20 epochs to see "clean" 256x256 crops to refine the bounding box accuracy for the 5x5 targets.
        multi_scale=False,  # This varies the input resolution (e.g., from $192$ to $320$) every few batches. It makes the kernels scale-invariant

        device = 0,  # Explicitly forces GPU 0
        project = "uav_detection",
        name = "stage2_full_tuning_v2"
    )

    final_weights = os.path.join("runs", "detect", "uav_detection", "stage2_full_tuning_v2", "weights", "best.pt")
    return final_weights


def export_openvino(final_weights_file):

    # =========================================================================
    # EXPORT STAGE: Hardware Export for Edge Computer
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPORTING: Converting Final Model to OpenVINO for Pi 5 CPU")
    print("=" * 60)

    final_model = YOLO(final_weights_file)

    # Export using FP16 precision for a significant processing boost on Pi 5
    final_model.export(format="openvino", imgsz=256, half=True)
    print("\nExport Complete! Your model is ready to deploy on your Raspberry Pi 5.")

# Apply the patch
metrics.bbox_iou = custom_bbox_iou
# ---------------------------

if __name__ == "__main__":
    yolo_model_cfg_file = '/home/roee/Projects/datasets/uav_vis_dataset/yolo_models/20260426_yolo26n_p2_spd_mixed/yolo26n_p2_spd_256x256.yaml'
    # dataset_yaml_file = '/home/roee/Projects/datasets/uav_vis_dataset/dataset_20260330/ultalytics_yolo_20260330/ultralytics_dataset_data.yaml'
    # dataset_yaml_file = '/home/roee/Projects/datasets/uav_vis_dataset/dataset_20260330/ultalytics_yolo_20260330_bg_balanced/ultralytics_dataset_data.yaml'
    # dataset_yaml_file = '/home/roee/Projects/datasets/uav_vis_dataset/dataset_20260429/ultalytics_yolo_20260429_bg_balanced/ultralytics_dataset_data.yaml'
    dataset_yaml_file = '/home/roee/Projects/datasets/uav_vis_dataset/dataset_20260513/ultalytics_yolo_20260513/ultralytics_dataset_data.yaml'

    pretrained_checkpoint = "yolo26n.pt"

    final_weights_file = train_yolo26_from_prior_weights(dataset_yaml_file, yolo_model_cfg_file, pretrained_checkpoint)
    export_openvino(final_weights_file)

    # final_weights_file = train_yolo26_from_scratch(dataset_yaml_file, yolo_model_cfg_file)
    # export_openvino(final_weights_file)

    # final_weights_file = train_yolo26_p2spd_v0(dataset_yaml_file, yolo_model_cfg_file)
    # export_openvino(final_weights_file)


    # Force evaluation on a completely pure, unseen test split
    # metrics = model.val(data="drone_dataset.yaml", split="test")
    # print(metrics.box.map50)  # Check if performance holds up here