import os
import shutil
import copy
import cv2
import numpy as np
import coco_dataset_export_for_training as cdeft


def get_random_crop_resize_test(draw_crop=False):
    full_img_size = (640, 480)
    required_image_size = (256, 256)
    annotation_bbox = [[300, 200, 70, 70],  # center
                       [0, 0, 100, 80],  # top left
                       [580, 50, 59, 40],  # top right
                       [50, 400, 100, 79],  # bottom left
                       [550, 400, 39, 79],  # bottom right
                       [320, 0, 80, 40],  # up center
                       [290, 400, 39, 79],  # bottom center
                       [0, 200, 120, 80],  # left center
                       [550, 230, 39, 30],  # right center
                       [350, 250, 80, 80]]  # center

    search_roi_scale_ratio = 2
    search_roi_min_width = 100
    search_roi_uncertainty_scale = (2, 5)

    n = 5

    # Load image
    img_file = './example_dataset/raw_dataset/images/0000394.png'
    img = cv2.imread(img_file)

    for ann_bbox in annotation_bbox:
        img_i = copy.deepcopy(img)

        # Draw annotation bbox
        cv2.rectangle(
            img_i,
            (ann_bbox[0], ann_bbox[1]),  # top-left
            (ann_bbox[0] + ann_bbox[2], ann_bbox[1] + ann_bbox[3]),  # bottom-right
            (0, 255, 0),  # color (B,G,R)
            2  # thickness
        )

        for i in range(n):
            crop_bbox, rescale = cdeft._get_random_crop_resize(full_img_size, ann_bbox, required_image_size,
                                                               search_roi_scale_ratio=search_roi_scale_ratio,
                                                               search_roi_min_width=search_roi_min_width,
                                                               search_roi_uncertainty_scale=search_roi_uncertainty_scale)
            print('crop {}: bbox={}, rescale={}'.format(i, crop_bbox, rescale))
            # check annotation is inside crop bbox
            assert crop_bbox[0] >= 0 and crop_bbox[0] + crop_bbox[2] - 1 <= full_img_size[0] - 1
            assert crop_bbox[1] >= 0 and crop_bbox[1] + crop_bbox[3] - 1 <= full_img_size[1] - 1
            # check crop bbox is inside the image
            assert crop_bbox[0] >= 0 and crop_bbox[0] + crop_bbox[2] - 1 <= full_img_size[0] - 1
            assert crop_bbox[1] >= 0 and crop_bbox[1] + crop_bbox[3] - 1 <= full_img_size[1] - 1

            # Draw crop bbox
            cv2.rectangle(
                img_i,
                (crop_bbox[0], crop_bbox[1]),  # top-left
                (crop_bbox[0] + crop_bbox[2], crop_bbox[1] + crop_bbox[3]),  # bottom-right
                (255, 0, 0),  # color (B,G,R)
                1  # thickness
            )

            if draw_crop:
                cv2.imshow("annotation {}".format(ann_bbox), img_i)
                cv2.waitKey(10)

                img_crop = cdeft._img_crop_resize(img_i, crop_bbox, required_image_size)

                # re - Draw crop annotation
                ann_crop = [ann_bbox[0] - crop_bbox[0], ann_bbox[1] - crop_bbox[1], ann_bbox[2], ann_bbox[3]]
                ann_crop = [int(np.round(x * rescale)) for x in ann_crop]
                cv2.rectangle(
                    img_crop,
                    (ann_crop[0], ann_crop[1]),  # top-left
                    (ann_crop[0] + ann_crop[2], ann_crop[1] + ann_crop[3]),  # bottom-right
                    (0, 0, 255),  # color (B,G,R)
                    1  # thickness
                )

                cv2.imshow("image crop {}".format(ann_bbox), img_crop)
                cv2.waitKey(0)

        # Show image
        cv2.imshow("annotation {}".format(ann_bbox), img_i)
        cv2.waitKey(10)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def images_resize_crop_test():
    # copy './example_dataset/raw_dataset' to './example_dataset/refined_dataset'
    source_dataset_folder = './example_dataset/raw_dataset'
    dataset_folder = './example_dataset/refined_dataset'
    if os.path.isdir(dataset_folder):
        shutil.rmtree(dataset_folder)
    shutil.copytreedd(source_dataset_folder, dataset_folder)

    input_coco_dataset_json = os.path.join(dataset_folder, 'annotations/coco_dataset.json')
    required_image_size = [256, 256]
    num_resizes = 5
    remove_original_annotated_images = False
    plot = True
    log_file = os.path.join(dataset_folder, 'refinery_log.txt')

    cdtr = cdeft.CocoDatasetRefinery(input_coco_dataset_json, log_file=log_file, verbose=True)
    cdtr.images_resize_crop(required_image_size, num_resizes, remove_original_annotated_images,
                           search_roi_scale_ratio=1, search_roi_min_width=50, search_roi_uncertainty_scale=(1, 5),
                           background_crop_scale=(1.5, 2.5))

    if plot:
        image_ids = cdtr.coco_dataset.get_image_ids()
        for img_id in image_ids:
            img = cdtr.coco_dataset.show_img(img_id, plot_annotations=True, color=(0, 255, 0), thickness=1)
            cv2.imshow("dataset images", img)
            cv2.waitKey(0)


if __name__ == '__main__':
    # get_random_crop_resize_test(draw_crop=True)
    images_resize_crop_test()

