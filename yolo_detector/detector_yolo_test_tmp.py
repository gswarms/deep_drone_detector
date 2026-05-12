""" Detect and track blobs in image
"""
import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO
from PIL import Image

if __name__ == '__main__':

    # trackign problem from close range - maybe try:
    # 1. force match to yolo_detector even with no overlap
    # 2. resize image so we track a blob with no shape! (in an adaptive way related to the shape of the target blob)

    record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_140609_extracted/images'
    frame_resize = (640, 480)

    use_cpu = False
    yolo_model_path = '../runs/detect/drone_detector_yolov8n/weights/best.pt'
    if os.path.isfile(yolo_model_path):
            yolo_model = YOLO('../runs/detect/drone_detector_yolov8n/weights/best.pt')
            if use_cpu:
                yolo_model.to('cpu')

    image_files = glob.glob(os.path.join(record_folder,'*.png'))
    image_files = sorted(image_files)

    # run on all frames
    tracking_active = False
    track = {'id': None, 'score': None, 'bbox': None}
    max_track_id = 0
    video_initialized = False
    frame_id = 0
    for image_file in image_files:
        img = cv2.imread(image_file)

        if frame_resize is not None:
            img = cv2.resize(img, frame_resize)


        # detect and track
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # results = yolo_model(img_rgb)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        results = yolo_model(img_pil)
        # results = yolo_model(img, imgsz=frame_resize[-1::-1])  # change imgsz

        res = []
        for r in results:
            for i, box in enumerate(r.boxes):
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                confidence = box.conf[0].item()  # Confidence score
                class_id = int(box.cls[0].item())  # Class index
                print(f"Class ID: {class_id}, Confidence: {confidence}, BBox: {xyxy}")
                res.append({'id': i, 'bbox': (xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]),
                            'confidence': confidence})

        # draw yolo_detector results
        img_to_draw = img.copy()
        for r in res:
            x, y, w, h = [int(v) for v in r['bbox']]
            img_to_draw = cv2.rectangle(img_to_draw, (x, y), (x + w, y + h), color=(50, 50, 255), thickness=1)
            cv2.putText(img_to_draw, '{}:{:.2f}'.format(r['id'], r['confidence']),
                        (x + w, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(100, 255, 255),
                        thickness=1)

        cv2.imshow('Frame', img_to_draw)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        frame_id = frame_id+1


    # Closes all the frames
    cv2.destroyAllWindows()
