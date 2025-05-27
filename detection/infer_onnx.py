import time
from ultralytics import YOLO
import cv2

# Load your saved model
model1 = YOLO('../runs/detect/drone_detector_yolov8n/weights/best.pt')
model1.to('cpu')

# Run inference
test_img_file = '/home/roee/Downloads/Drone Dataset.v2i.yolov8/test/images/camera_2025_4_8-12_40_29_1744116048841747231_png.rf.48755d87ba0560dcd7d20848affaf73f.jpg'

img = cv2.imread(test_img_file)



# image size Vs time tests
# --------------------------------------------------------
img_resize_shape = [(640, 480),
                    (480, 352),
                    (320, 256)]

for i in range(len(img_resize_shape)):
    img_resize_sh = img_resize_shape[i]
    img_resized = cv2.resize(img, img_resize_sh)
    results = model1(img_resized, imgsz=img_resize_sh[-1::-1])  # change imgsz
    t1 = time.monotonic()
    for i in range(40):
        results = model1(img_resized, imgsz=img_resize_sh[-1::-1])  # change imgsz
    t2 = time.monotonic()
    print('-----------------------')
    print('avarage ({}x{})) inference time = {}[sec]'.format(img_resize_sh[0], img_resize_sh[1],(t2-t1)/40))




# Run inference
# --------------------------------------------------------
model2 = YOLO('../runs/detect/drone_detector_yolov8n/weights/best.pt')
results = model2.predict(test_img_file, imgsz=(320, 240))
test_imgs = test_img_file * 10
t1 = time.monotonic()
for i in range(40):
    results = model2.predict(test_img_file)
t2 = time.monotonic()
print('avarage inference time = {}[sec]'.format((t2-t1)/40))


for r in results:# or .predict(...)
    results.show()
