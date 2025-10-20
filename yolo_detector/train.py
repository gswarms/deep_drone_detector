import os.path

from ultralytics import YOLO
# from ultralytics.data.loaders import build_dataloader
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

# Load a YOLOv8n model (nano)
# model = YOLO('../models/yolov8n.pt')
model = YOLO('../models/yolo11n.pt')




# Train the model
print('--------------------------- model train ----------------------')
# model.train(
#     data='/home/roee/Downloads/Drone Dataset.v2i.yolov8/data.yaml',
#     epochs=50,
#     imgsz=(320, 240),
#     batch=16,
#     name='drone_detector_yolov8n'
# )


dataset_yaml_file = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/yolo_dataset.yaml'

# cfg = get_cfg()
# cfg.data = dataset_yaml_file
# check_det_dataset(cfg)

# output_name = 'drone_detector_yolov8n_20250709'
output_name = 'drone_detector_yolov11n_320x240_20251021'

model.train(
    data=dataset_yaml_file,
    epochs=100,
    imgsz=(320, 240),
    batch=16,
    freeze=10,
    fliplr=0.5,
    name=output_name
)



print('--------------------------- model evaluation ----------------------')
metrics = model.val()
print(metrics)



# print('--------------------------- test results ----------------------')
#
# results = model.predict(source='sample.jpg', save=True, conf=0.25)
# model.predict(source='images/test/', save=True)
# results.show()    # Inline display
# results.save()    # Saves to runs/predict/




