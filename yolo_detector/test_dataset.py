""" Run model on dataset, and analise results
"""
import os
import cv2
import numpy as np
import glob


if __name__ == '__main__':


    coco_dataset_file = ''


    # load coco dataset
    if not os.path.isfile(coco_dataset_file):
        raise Exception('coco dataset file not found: {}'.format(coco_dataset_file))
    import coco_dataset_utils as cdu
    dataset = cdu.CocoDatasetManager()
    dataset.load_coco(coco_dataset_file)


    # infer and compare to dataset labels


    # trackign problem from close range - maybe try:
    # 1. force match to yolo_detector even with no overlap
    # 2. resize image so we track a blob with no shape! (in an adaptive way related to the shape of the target blob)

    record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_140609_extracted/images'
    frame_resize = (640, 480)

    # run on all frames
    image_files = glob.glob(os.path.join(record_folder,'*.png'))
    image_files = sorted(image_files)

    frame_id = 0
    for image_file in image_files:
        img = cv2.imread(image_file)
        img = cv2.resize(img, frame_resize)
        cv2.imshow('Frame', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        frame_id = frame_id+1


    # Closes all the frames
    cv2.destroyAllWindows()
