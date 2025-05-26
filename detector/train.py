from ultralytics import YOLO

# Load a YOLOv8n model (nano)
model = YOLO('../yolov8n.pt')



# Train the model
print('--------------------------- model train ----------------------')
model.train(
    data='/home/roee/Downloads/Drone Dataset.v2i.yolov8/data.yaml',
    epochs=50,
    imgsz=(320, 240),
    batch=16,
    name='drone_detector_yolov8n'
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




