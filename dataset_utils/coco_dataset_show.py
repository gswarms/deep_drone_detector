import os
import pathlib
import numpy as np
import cv2
import coco_dataset_manager


def save_to_video(dataset_json_file, roi_size, output_video_file, video_fps=20):
    """
    merge single experiment scenario datasets to one big dataset

    :param dataset_json_file:
    :param roi_size: (width, height)
    :param output_video_file:
    :param video_fps:
    """

    # load dataset
    dataset_json_file = pathlib.Path(dataset_json_file)
    print('loading dataset from: {}'.format(dataset_json_file))
    if not dataset_json_file.exists():
        raise Exception('dataset json file: {} not found!'.format(dataset_json_file))
    dataset = coco_dataset_manager.CocoDatasetManager()
    dataset.load_coco(dataset_json_file, verify_image_files=False)

    # Define the codec and create VideoWriter object
    # 'mp4v' is a common codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_file, fourcc, video_fps, roi_size)

    # go over all images
    image_ids_annotated = set(dataset.df_annotations['image_id'])  # sorted
    target_range = {0: 200, 88:2}  # {frame_id: range}
    for img_id in image_ids_annotated:
        img_data = dataset.get_image(img_id)
        annotations_data = dataset.get_image_annotations(img_id)

        img_path = os.path.join(dataset.images_folder, img_data['file_name'])
        img = cv2.imread(img_path)
        ann_bbox = annotations_data[0]['bbox']

        cx = ann_bbox[0] + ann_bbox[2]/2
        cy = ann_bbox[1] + ann_bbox[3]/2

        xtl = max(int(np.round(cx - roi_size[0] / 2)), 0)
        ytl = max(int(np.round(cy - roi_size[1] / 2)), 0)
        img_roi = img[ytl:ytl + roi_size[1], xtl:xtl + roi_size[0], :]

        img_roi = cv2.resize(img_roi, (roi_size[0], roi_size[1]))

        # Define text properties
        text_str = "fid {} d={}".format(img_id, 101)
        position = (20, 20)  # (x, y) coordinates of the bottom-left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        color = (255, 0, 0)  # Blue in BGR
        thickness = 1
        # Apply text to image
        cv2.putText(img_roi, text_str, position, font, font_scale, color, thickness, cv2.LINE_AA)

        cv2.imshow('Roi Image', img_roi)
        cv2.waitKey(20)

        video.write(img_roi)


    # Release the video writer
    cv2.destroyAllWindows()
    video.release()
    cv2.destroyAllWindows()
    print(f"Video saved successfully as {output_video_file}")





if __name__ == '__main__':

    dataset_folder = '/home/roee/Projects/datasets/interceptor_drone/uav_detection_dataset/scenarios_vis/20260420_reshafim/20260420_1256_13/20260420_1301_16/camera_20260420_1301_extracted'
    dataset_json_file = os.path.join(dataset_folder, 'annotations', 'coco_dataset.json')
    output_video_file = os.path.join(dataset_folder, 'target_vzoom_video.avi')

    roi_size = (120, 80)
    save_to_video(dataset_json_file, roi_size, output_video_file)

    print('Done!')