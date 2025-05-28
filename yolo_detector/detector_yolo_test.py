""" Detect and track blobs in image
"""
import os
import cv2
import sys
import numpy as np
from test_utils.roi_utils import PolygonPerFrame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test_utils.standard_record import StandardRecord
from ultralytics import YOLO

if __name__ == '__main__':

    # trackign problem from close range - maybe try:
    # 1. force match to yolo_detector even with no overlap
    # 2. resize image so we track a blob with no shape! (in an adaptive way related to the shape of the target blob)

    record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_140609_extracted'
    # frame_size = (640, 480)
    frame_size = (640, 480)
    frame_resize = None
    start_time = -np.inf
    blob_detection_params_file = '/home/roee/Projects/blob_detector/examples/blob_detector_params_20250511_kfar_galim.yaml'
    valid_roi = None
    output_video_file = os.path.join(record_folder, '20250511_140609_kfar_galim_results.avi')
    polygons_file = os.path.join(record_folder, 'kfar_galim_20250511_140609_polygons.yaml')  # ***optional


    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_133041_extracted'
    # frame_size = (640, 480)
    # frame_resize = None
    # start_time = -np.inf
    # blob_detection_params_file = '/home/roee/Projects/blob_detector/examples/blob_detector_params_20250511_kfar_galim.yaml'
    # valid_roi = None
    # output_video_file = os.path.join(record_folder, '20250511_133041_kfar_galim_results.avi')
    # polygons_file = os.path.join(record_folder, 'kfar_galim_20250511_133041_polygons.yaml')  # ***optional

    use_cpu = False
    yolo_model_path = '../runs/detect/drone_detector_yolov8n/weights/best.pt'
    if os.path.isfile(yolo_model_path):
            yolo_model = YOLO('../runs/detect/drone_detector_yolov8n/weights/best.pt')
            if use_cpu:
                yolo_model.to('cpu')

    # get roi polygons per frame
    if polygons_file is not None:
        polygons_per_frame = PolygonPerFrame(frame_resize)
        polygons_per_frame.load(polygons_file)
        if (polygons_per_frame.frame_size[0] != frame_size[0] or
                polygons_per_frame.frame_size[1] != frame_size[1]):
            raise Exception('frame size does not fit polygons_per_frame frame size!')
    else:
        polygons_per_frame = None

    # get record
    record = StandardRecord(record_folder)

    # run on all frames
    tracking_active = False
    track = {'id': None, 'score': None, 'bbox': None}
    max_track_id = 0
    video_initialized = False
    frame_id = 0
    for frame in record.frames:
        if frame['time'] >= start_time:
            img = cv2.imread(frame['image_file'])

            if frame_resize is not None:
                img = cv2.resize(img, frame_resize)

            roi_polygon = None
            if polygons_per_frame is not None:
                roi_polygon = polygons_per_frame.get(frame_id)
                # if roi_polygon is not None:
                #     dttr.set_detection_roi_polygon(roi_polygon, method='crop')

            # detect and track
            results = yolo_model(img)
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

            # draw roi
            img_to_draw = img
            if roi_polygon is not None:
                poly = np.floor(roi_polygon).astype(np.int32)
                img_to_draw = cv2.polylines(img, [poly], isClosed=True, color=(255, 50, 50), thickness=1)

            # draw yolo_detector results
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
