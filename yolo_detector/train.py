import os.path
import time
from ultralytics import YOLO
# from ultralytics.data.loaders import build_dataloader
# from ultralytics.cfg import get_cfg
# from ultralytics.data.utils import check_det_dataset

# Load a YOLOv8n model (nano)
# model = YOLO('../models/yolov8n.pt')
# model = YOLO('../models/yolo11n.pt')
# model = YOLO('../models/yolo26n.pt')




# Train the model
print('--------------------------- model train ----------------------')
# model.train(
#     data='/home/roee/Downloads/Drone Dataset.v2i.yolov8/data.yaml',
#     epochs=50,
#     imgsz=(320, 240),
#     batch=16,
#     name='drone_detector_yolov8n'
# )

t0 = time.monotonic()
# dataset_yaml_file = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/yolo_dataset.yaml'
# dataset_yaml_file = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/yolo_dataset.yaml'
# dataset_yaml_file = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/ultalytics_yolo_20260104/ultralytics_dataset_data.yaml'
# dataset_yaml_file = '/home/roee/Projects/datasets/UAV_fixed_wing_dataset/dataset_20260323/ultalytics_yolo_20260324/ultralytics_dataset_data.yaml'
dataset_yaml_file = '/home/roee/Projects/datasets/UAV_fixed_wing_dataset/dataset_20260330/ultalytics_yolo_20260330/ultralytics_dataset_data.yaml'

# cfg = get_cfg()
# cfg.data = dataset_yaml_file
# check_det_dataset(cfg)


# output_name = 'drone_detector_yolov8n_20250709'
# output_name = 'drone_detector_yolov11n_320x240_20251021'
# output_name = 'drone_detector_yolov11n_256x256_20260104'

# ----------------------------- yolo26n p2 model ---------------------------------------
output_name = 'drone_detector_yolov26n_256x256_20260324_mixed_p2_v1'
yolo26_p2_yaml_file = '/home/roee/Projects/datasets/UAV_fixed_wing_dataset/dataset_20260330/ultalytics_yolo_20260330/yolo26n-p2.yaml'
yolo26_model = '/home/roee/Projects/datasets/UAV_fixed_wing_dataset/yolo_models/yolo26n.pt'
model = YOLO(yolo26_p2_yaml_file).load(yolo26_model)

# output_name = 'drone_detector_yolov26n_256x256_20260324_mixed_v1'
# model = YOLO('../models/yolo26n.pt')

model.train(
    data=dataset_yaml_file,
    epochs=250,
    imgsz=256,
    batch=32,
    multi_scale=False,    # FIX: Change to False to prevent the ZeroDivisionError
    rect=True,            # Optimized for fixed-size small images
    copy_paste=0.5,       # Keep this to help with the large-drone imbalance
    mosaic=1.0,
    optimizer='MuSGD',
    box=10.0,
    cls=1.5,
    patience=50,
    mixup=0.1
)


# model = YOLO('../models/yolo26n.pt')
# output_name = 'drone_detector_yolov26n_256x256_20260324_mixed'
# model.train(
#     data=dataset_yaml_file,
#     imgsz=256,
#     epochs=250,
#     batch=16,
#     freeze=5,
#     optimizer="AdamW",
#     augment=True,
#     mosaic=0.5,        # Lowered to keep focus on centered targets
#     close_mosaic=15,
#     multi_scale=False,  # If inference is strictly 256, keep training strictly 256
#     scale=0.5,         # 0.0-0.3 Allows the model to see very small distant drones
#     fliplr=0.5,
#     flipud=0.2,        # Drones can be seen from below/above in dogfights
#     project="runs/train",
#     name=output_name,
#     cache=True,
#     device=0
# )

print('training run time: {}[sec]'.format(time.monotonic()-t0))

# --------------- 20251211 ---------------------------
# model.train(
#     data=dataset_yaml_file,
#     imgsz=256,
#     epochs=250,
#     batch=16,
#     freeze=5,
#     optimizer="AdamW",
#     augment=True,
#     mosaic=0.5,        # Lowered to keep focus on centered targets
#     close_mosaic=15,
#     multi_scale=False,  # If inference is strictly 256, keep training strictly 256
#     scale=0.5,         # 0.0-0.3 Allows the model to see very small distant drones
#     fliplr=0.5,
#     flipud=0.2,        # Drones can be seen from below/above in dogfights
#     project="runs/train",
#     name=output_name,
#     cache=True,
#     device=0
# )


# ----------------------------- older ---------------------------------
# model.train(
#     data=dataset_yaml_file,
#     imgsz=256,
#     epochs=150,
#     batch=16,           # auto
#     freeze=5,  # freeze first 10 layers (rough example)
#     rect=False,  # Use rect=True to reduce padding and preserve aspect ratio. (not good for small objects)
#     optimizer="AdamW", # "AdamW" for small dataset. "" for large dataset
#     augment=True,
#     mosaic=1.0,  # default ON
#     close_mosaic=10, # disable last 10 epochs
#     multi_scale=True,  # critical for small objects
#     scale=0.7,  # scale < 1 allows UPSCALING small drones
#     # translate=0.1,
#     cache=True,
#     workers=8,
#     device=0,  # GPU
#     project="runs/train",
#     name=output_name
#     )

# model.train(
#     data=dataset_yaml_file,
#     epochs=100,
#     imgsz=(256, 256),
#     batch=16,
#     freeze=10,
#     augment=True,
#     fliplr=0.5,
#     name=output_name,
#     device=0,  # GPU
#     mosaic=1.0,  # default ON
#     close_mosaic=10  # disable last 10 epochs
# )



print('--------------------------- model evaluation ----------------------')
metrics = model.val()
print(metrics)



# print('--------------------------- test results ----------------------')
#
# results = model.predict(source='sample.jpg', save=True, conf=0.25)
# model.predict(source='images/test/', save=True)
# results.show()    # Inline display
# results.save()    # Saves to runs/predict/




