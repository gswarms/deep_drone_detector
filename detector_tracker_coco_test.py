""" Detect and track blobs in image
"""
import os
import cv2
import sys
import numpy as np
import re
from pathlib import Path
from dataset_utils import coco_dataset_manager
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import detector_tracker


if __name__ == '__main__':

    # ------------------ kfar galim 27.10.2025 ------------------------------
    dataset_json_file = '/home/roee/Projects/datasets/interceptor_drone/20251027_kfar_galim/20251027_123000/camera_20251027_1230_extracted'

    frame_size = (640, 480)
    frame_resize = None

    # -------------------- yolov8n -------------
    # model_path = 'runs/detect/drone_detector_yolov8n/weights/best.pt'
    # detection_frame_size = (640, 480)
    # detection_frame_size = (320, 320)
    # detection_frame_size = (256, 256)
    # detection_frame_size = (224, 224)
    # detector_type = 'yolov8n'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov8n/weights/best_320.onnx'
    # detection_frame_size = (320, 320)
    # detector_type = 'yolov8n'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov8n/weights/best_256.onnx'
    # model_path = '/home/roee/Projects/deep_drone_detector/runs/detect/20250709_drone_detector_yolov8n3/weights/best_256.onnx'
    # detection_frame_size = (256, 256)
    # detector_type = 'yolov8n'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov8n/weights/best_224.onnx'
    # detection_frame_size = (224, 224)
    # detector_type = 'yolov8n'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'
    # config_path = None

    # -------------------- yolov11n -------------
    # model_path = 'runs/detect/drone_detector_yolov11n_320x240_20251021/weights/best.pt'
    # detection_frame_size = (320, 320)
    # detector_type = 'yolov11n'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov11n_320x240_20251021/weights/yolov11n_best_lulav_dit_320x320.onnx'
    # detection_frame_size = (320, 320)
    # detector_type = 'yolov11n'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov11n_320x240_20251021/weights/yolov11n_best_lulav_dit_256x256.onnx'
    # detection_frame_size = (256, 256)
    # detector_type = 'yolov11n'
    # config_path = None

    model_path = 'runs/detect/drone_detector_yolov11n_320x240_20251021/weights/yolov11n_best_lulav_dit_256x256_openvino.xml'
    config_path = 'runs/detect/drone_detector_yolov11n_320x240_20251021/weights/yolov11n_best_lulav_dit_256x256_openvino.bin'
    detection_frame_size = (256, 256)
    detector_type = 'yolov11n'

    step_mode = 'detect_only'  # 'detect_only' / 'detect_track' / 'track_only'

    # set video writer
    dataset_folder = Path(dataset_json_file).parent.parent
    now = datetime.now()
    output_video_file = os.path.join(dataset_folder, 'results_' + now.strftime("%Y%m%d_%H%M%S") +'.avi')
    if output_video_file is not None:
        print('saving record video to: {}'.format(output_video_file))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_file, fourcc, 20, (640, 480))

    # setup detector tracker
    dttr = detector_tracker.DetectorTracker(model_path, detection_frame_size, detector_type=detector_type,
                                            bbox_roi_intersection_th=0.1, detector_use_cpu=False, verbose=False,
                                            detector_config_path=config_path)

    # load dataset
    dataset = coco_dataset_manager.CocoDatasetManager()
    dataset.load_coco(dataset_json_file, verify_image_files=True)

    # run on all frames
    tracking_active = False
    track = {'id': None, 'score': None, 'bbox': None}
    max_track_id = 0
    video_initialized = False
    frame_id = 0
    tr = []
    img_ids = dataset.get_image_ids()
    for img_id in img_ids:
        # get image
        img_data = dataset.get_image(img_id)
        img = cv2.imread(img_data['image_file'])

        # get reference annotations
        annotations_data = dataset.get_image_annotations(img_id)

        # detect and track
        if step_mode == 'detect_only':
            tr = dttr.step(img, conf_threshold=0.5, nms_iou_threshold=0.4, max_num_detections=10,
                           operation_mode='detect_only')
        elif step_mode == 'detect_track':
            tr = dttr.step(img, conf_threshold=0.5, nms_iou_threshold=0.4, max_num_detections=10,
                           operation_mode='detect_track')
        elif step_mode == 'track_only':
            # detect once, and then track only
            if len(tr)>0:
                tr = dttr.step(img, conf_threshold=0.5, nms_iou_threshold=0.4, max_num_detections=10, operation_mode='track_only')
            else:
                tr = dttr.step(img, conf_threshold=0.5, nms_iou_threshold=0.4, max_num_detections=10,
                           operation_mode='detect_track')

        # compare results
        #TODO: compare results

        # draw
        # TODO: draw good / bad / missed detections in different colors
        img_to_draw = dttr.draw(img)
        img_to_draw = cv2.putText(img_to_draw, '{:d}'.format(frame_id), (20, 20),
                                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(50, 100,200), thickness=2)
        cv2.imshow('Frame', img_to_draw)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        if (output_video_file is not None) and (not video_initialized):
            print('saving record video to: {}'.format(output_video_file))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_file, fourcc, 20, (int(img.shape[1]), int(img.shape[0])))
            video_initialized = True

        if output_video_file is not None:
            out.write(img_to_draw)

        frame_id = frame_id+1


    # Closes all the frames
    cv2.destroyAllWindows()

    if output_video_file is not None:
        out.release()