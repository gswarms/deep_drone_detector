import copy
import os
import shutil
import random

import cv2
import numpy as np
from collections import defaultdict
import yaml
import coco_dataset_utils
import dataset_training_utils



def export_for_training(coco_dataset_json_file, output_dir, split_ratios={'train': 0.7, 'val': 0.15, 'test':0.15},
                        augment_crop_size=None, augment_crop_number=None):
    """
    make coco dataset ready for yolo ultralytics training.

    dataset file structure:

    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── annotations/
    │   ├── instances_train.json
    │   ├── instances_val.json
    │   └── instances_test.json

    additional

    :param coco_dataset_json_file: path to coco json file
    :param output_dir: path train-ready dataset folder
    :param augment_crop_size: size of augment-crop (w, h)
                              images will be randomly cropped to augment_crop_size, and then resized to the original image size.
                              annotated bbox will be adjusted accordingly
                              None means no augment-crops
    :param augment_crop_number: number of crop augmentations for each image
    :return:
    """

    # handle inputs
    if (augment_crop_size is None and augment_crop_number is not None) or (augment_crop_size is not None and augment_crop_number is None):
        raise Exception('invalid augment_crop parameters')

    #---------------------- make dataset file structure ready ----------------------
    output_images_folder = os.path.join(output_dir, 'images')
    if not os.path.isdir(output_images_folder):
        os.makedirs(output_images_folder)
    output_images_train_folder = os.path.join(output_images_folder, 'train')
    if not os.path.isdir(output_images_train_folder):
        os.makedirs(output_images_train_folder)
    output_images_val_folder = os.path.join(output_images_folder, 'val')
    if not os.path.isdir(output_images_val_folder):
        os.makedirs(output_images_val_folder)
    output_images_test_folder = os.path.join(output_images_folder, 'test')
    if not os.path.isdir(output_images_test_folder):
        os.makedirs(output_images_test_folder)

    output_annotations_folder = os.path.join(output_dir, 'annotations')
    if not os.path.isdir(output_annotations_folder):
        os.makedirs(output_annotations_folder)
    output_annotations_train_file = os.path.join(output_annotations_folder, 'instances_train.json')
    output_annotations_val_file = os.path.join(output_annotations_folder, 'instances_val.json')
    output_annotations_test_file = os.path.join(output_annotations_folder, 'instances_test.json')

    # make output dataset yolo yaml file
    output_dataset_yolo_yaml_file = os.path.join(output_dir, 'data.yaml')


    #---------------------- load dataset -----------------------
    coco_dataset = coco_dataset_utils.CocoDatasetManager()
    coco_dataset.load(coco_dataset_json_file, verify_images = True)


    #----------------- split to train / test / val sets -------------------
    # get all annotated images
    image_ids = coco_dataset.get_annotated_image_ids()

    # split dataset
    # 0=train, 1=val, 2=test
    n = len(image_ids)
    dataset_types = np.random.choice((0,1,2), size=n, p=(split_ratios['train'], split_ratios['val'], split_ratios['test']))

    coco_train_dataset = coco_dataset_utils.CocoDatasetManager()
    coco_val_dataset = coco_dataset_utils.CocoDatasetManager()
    coco_test_dataset = coco_dataset_utils.CocoDatasetManager()
    coco_dataset_origin_path = os.path.dirname(coco_dataset_json_file)
    for i, dataset_type in enumerate(dataset_types):

        # get image data
        img_data = coco_dataset.get_image(image_id=image_ids[i])
        img_record = coco_dataset_utils.ImageRecord()
        img_record.from_dict(img_data)

        # copy image to output folder
        img_abs_path = img_record.file_name
        if dataset_type == 0:
            img_dst_path = os.path.join(output_images_train_folder, os.path.basename(img_record.file_name))
        elif dataset_type == 1:
            img_dst_path = os.path.join(output_images_val_folder, os.path.basename(img_record.file_name))
        elif dataset_type == 2:
            img_dst_path = os.path.join(output_images_test_folder, os.path.basename(img_record.file_name))
        else:
            raise Exception('invalid dataset index!')
        shutil.copy(img_abs_path, img_dst_path)

        # convert relative path considering the new annotations file and image file
        img_relative_path = os.path.relpath(img_dst_path, output_annotations_folder)
        img_record.file_name = img_relative_path

        # get annotations
        img_annotations_dict = coco_dataset.get_annotations(img_record.id)
        if img_annotations_dict is not None:
            img_annotations = []
            for iad in img_annotations_dict:
                img_ann = coco_dataset_utils.Annotation()
                img_ann.from_dict(iad)
                img_annotations.append(img_ann)
        else:
            img_annotations = None

        img_records =  [img_record]
        img_annotations_records = [img_annotations]
        bboxes_orig = [x.bbox for x in img_annotations]
        img_annotations_orig = img_annotations

        # augment-crop images
        if augment_crop_number is not None and augment_crop_number > 0:
            img = cv2.imread(img_dst_path)
            sp = os.path.basename(img_dst_path).split('.')
            pth = os.path.dirname(img_dst_path)
            for j in range(augment_crop_number):
                img_cropped, bboxes_cropped_i = dataset_training_utils.augment_crop_image(img, augment_crop_size,
                                                      bboxes=bboxes_orig, resize_to_original=True)
                cropped_img_file = os.path.join(pth, sp[0]+'_aug{}.'.format(j)+sp[1])
                cv2.imwrite(cropped_img_file, img_cropped)
                img_record_cropped = copy.deepcopy(img_record)
                img_record_cropped.file_name = cropped_img_file
                img_records.append(img_record_cropped)

                if img_annotations_orig is not None:
                    img_annotations_cropped = copy.deepcopy(img_annotations_orig)
                    for k, irc in enumerate(img_annotations_cropped):
                            irc.bbox = bboxes_cropped_i[k]
                    img_annotations_records.append(img_annotations_cropped)

        # add to relevant dataset
        for j in range(len(img_records)):
            if dataset_type == 0:
                new_img_id = coco_train_dataset.add_image(img_records[j].file_name, img_records[j].width, img_records[j].height,
                                                          img_records[j].scenario, img_records[j].date, img_records[j].daytime,
                                                          img_records[j].weather, img_records[j].cloud_coverage)
                if img_annotations_records[j] is not None:
                    for ann in img_annotations_records[j]:
                        if ann.bbox is not None:
                            coco_train_dataset.add_annotation(new_img_id, ann.category_id, ann.bbox,
                                                iscrowd=ann.is_crowd,
                                                distance_from_camera=ann.distance_from_camera)

            elif dataset_type == 1:
                new_img_id = coco_val_dataset.add_image(img_records[j].file_name, img_records[j].width, img_records[j].height,
                                                          img_records[j].scenario, img_records[j].date, img_records[j].daytime,
                                                          img_records[j].weather, img_records[j].cloud_coverage)
                if img_annotations_records[j] is not None:
                    for ann in img_annotations_records[j]:
                        if ann.bbox is not None:
                            coco_val_dataset.add_annotation(new_img_id, ann.category_id, ann.bbox,
                                                iscrowd=ann.is_crowd,
                                                distance_from_camera=ann.distance_from_camera)

            elif dataset_type == 2:
                new_img_id = coco_test_dataset.add_image(img_records[j].file_name, img_records[j].width, img_records[j].height,
                                                          img_records[j].scenario, img_records[j].date, img_records[j].daytime,
                                                          img_records[j].weather, img_records[j].cloud_coverage)
                if img_annotations_records[j] is not None:
                    for ann in img_annotations_records[j]:
                        if ann.bbox is not None:
                            coco_test_dataset.add_annotation(new_img_id, ann.category_id, ann.bbox,
                                                iscrowd=ann.is_crowd,
                                                distance_from_camera=ann.distance_from_camera)

            else:
                raise Exception('invalid dataset index!')



    # copy categories
    coco_train_dataset.categories = copy.deepcopy(coco_dataset.get_categories())
    coco_val_dataset.categories = copy.deepcopy(coco_dataset.get_categories())
    coco_test_dataset.categories = copy.deepcopy(coco_dataset.get_categories())

    # save
    coco_train_dataset.save(output_annotations_train_file)
    coco_val_dataset.save(output_annotations_val_file)
    coco_test_dataset.save(output_annotations_test_file)


    #----------------- make output dataset yolo yaml file -------------------
    cat = []
    for i, c in enumerate(coco_dataset.categories):
        if c['id'] != i+1:
            raise Exception('unordered categories')
        cat.append(c['name'])

    data = {'train': output_annotations_train_file,
            'val': output_annotations_val_file,
            'test': output_annotations_test_file,
            'nc': len(cat),
            'names': cat
            }
    with open(output_dataset_yolo_yaml_file, 'w') as f:
        yaml.dump(data, f, sort_keys=False)


if __name__ == '__main__':
    baseline_dataset_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/coco_dataset.json'
    res_dataset_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco'

    # CONFIG
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    seed = 42
    random.seed(seed)

    # run
    export_for_training(baseline_dataset_folder, res_dataset_folder,
                        split_ratios=split_ratios, augment_crop_size=(320, 240), augment_crop_number=3)