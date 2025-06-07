""" Detect and track blobs in image
"""
import os
import cv2
import sys
import numpy as np
from test_utils.roi_utils import PolygonPerFrame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import detector_tracker
from test_utils.standard_record import StandardRecord

if __name__ == '__main__':

    # trackign problem from close range - maybe try:
    # 1. force match to yolo_detector even with no overlap
    # 2. resize image so we track a blob with no shape! (in an adaptive way related to the shape of the target blob)

    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_140609_extracted'
    # frame_size = (640, 480)
    # frame_resize = None
    # start_time = -np.inf
    # output_video_file = os.path.join(record_folder, '20250511_140609_kfar_galim_results.avi')
    # polygons_file = os.path.join(record_folder, 'kfar_galim_20250511_140609_polygons.yaml')  # ***optional


    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_133041_extracted'
    # frame_size = (640, 480)
    # frame_resize = None
    # start_time = -np.inf
    # output_video_file = os.path.join(record_folder, '20250511_133041_kfar_galim_results.avi')
    # polygons_file = os.path.join(record_folder, 'kfar_galim_20250511_133041_polygons.yaml')  # ***optional


    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250519_kfar_galim/camera_20250519_083827_extracted'
    # frame_size = (640, 480)
    # frame_resize = None
    # start_time = -np.inf
    # output_video_file = os.path.join(record_folder, '20250519_083827_kfar_galim_results.avi')
    # polygons_file = os.path.join(record_folder, 'kfar_galim_20250519_083827_polygons.yaml')  # ***optional


    record_folder = '/home/roee/Downloads/camera_2025_6_5-12_56_26_extracted'
    frame_size = (640, 480)
    frame_resize = None
    start_time = -np.inf
    output_video_file = os.path.join(record_folder, '20250605_125626_kfar_galim_results.avi')
    polygons_file = os.path.join(record_folder, 'kfar_galim_20250605_125626_polygons.yaml')  # ***optional


    # yolo_model_path = 'runs/detect/drone_detector_yolov8n/weights/best.pt'
    yolo_model_path = 'runs/detect/drone_detector_yolov8n/weights/best_320.onnx'

    # set video writer
    if output_video_file is not None:
        print('saving record video to: {}'.format(output_video_file))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_file, fourcc, 20, (640, 480))

    # setup blob tracker
    detection_frame_size = (320, 320)
    # detection_frame_size = (640, 480)
    dttr = detector_tracker.DetectorTracker(yolo_model_path, detection_frame_size,
                                            bbox_roi_intersection_th=0.1, detector_use_cpu=False, verbose=False)

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

            if polygons_per_frame is not None:
                roi_polygon = polygons_per_frame.get(frame_id)
                if roi_polygon is not None:
                    dttr.set_detection_roi_polygon(roi_polygon, method='crop')

            if (output_video_file is not None) and (not video_initialized):
                print('saving record video to: {}'.format(output_video_file))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_video_file, fourcc, 20, (int(img.shape[1]), int(img.shape[0])))
                video_initialized = True

            # detect and track
            if frame_id == 41:
                aa=5
            tr = dttr.step(img, conf_threshold=0.5, nms_iou_threshold=0.4, max_num_detections=10)
            img_to_draw = dttr.draw(img)
            img_to_draw = cv2.putText(img_to_draw, '{:d}'.format(frame_id), (20, 20),
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(50, 100,200), thickness=2)
            # draw
            cv2.imshow('Frame', img_to_draw)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if output_video_file is not None:
                out.write(img_to_draw)

            if len(tr)>0:  # debug - find first yolo_detector
                aa=5
            frame_id = frame_id+1


    # Closes all the frames
    cv2.destroyAllWindows()

    if output_video_file is not None:
        out.release()