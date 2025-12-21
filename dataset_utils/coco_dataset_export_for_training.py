import copy
import os
import shutil
import random

import cv2
import numpy as np
import yaml
import coco_dataset_manager
import dataset_training_utils



def export_for_training(coco_dataset_json_file, output_dir, split_ratios={'train': 0.7, 'val': 0.15, 'test':0.15},
                        augment_crop_size=None, augment_crop_number=None):
    """
    export coco dataset for yolo ultralytics training format.

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

    output yolo dataset structure:
    yolo_output_dir/
    ├── yolo_data.yaml
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train
    │   │   ├── <imgid 1>.txt
    │   │   ├── ...
    │   │   ├── <imgid n>.txt
    │   ├── val
    │   │   ├── <imgid n+1>.txt
    │   │   ├── ...
    │   │   ├── <imgid m>.txt
    │   ├── test
    │   │   ├── <imgid m+1>.txt
    │   │   ├── ...
    │   │   ├── <imgid k>.txt

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
    coco_dataset.load_coco(coco_dataset_json_file, verify_images = True)

    print('loaded coco dataset from: {}'.format(coco_dataset_json_file))
    coco_dataset.verify()

    #----------------- split to train / test / val sets -------------------
    # get all annotated images
    # image_ids = [x['id'] for x in coco_dataset.image_records]  # TODO: augment all images - with ro without annotations
    image_ids = coco_dataset.get_image_ids()

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
        # img_relative_path = os.path.relpath(img_dst_path, output_annotations_folder)
        # img_record.file_name = img_relative_path
        img_record.file_name = img_dst_path

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
        if img_annotations is not None:
            bboxes_orig = [x.bbox for x in img_annotations]
        else:
            bboxes_orig = None
        img_annotations_orig = img_annotations

        # augment-crop images
        if augment_crop_number is not None and augment_crop_number > 0:
            img = cv2.imread(img_dst_path)
            sp = os.path.basename(img_dst_path).split('.')
            pth = os.path.dirname(img_dst_path)
            for j in range(augment_crop_number):
                img_cropped, bboxes_cropped_i = dataset_training_utils.augment_crop_image(img, augment_crop_size,
                                                      bboxes=bboxes_orig, resize_to_original=True, min_valid_bbox_crop_ratio=0.3)
                cropped_img_file = os.path.join(pth, sp[0]+'_aug{}.'.format(j)+sp[1])
                cv2.imwrite(cropped_img_file, img_cropped)
                # cropped_img_relative_path = os.path.relpath(cropped_img_file, output_annotations_folder)
                img_record_cropped = copy.deepcopy(img_record)
                img_record_cropped.file_name = cropped_img_file
                img_records.append(img_record_cropped)

                if img_annotations_orig is None:
                    img_annotations_records.append(None)
                else:
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

    print('verifying train dataset:')
    coco_train_dataset.verify()
    print('verifying val dataset:')
    coco_val_dataset.verify()
    print('verifying test dataset:')
    coco_test_dataset.verify()


    # convert to analytics-yolo format
    output_labels_train_dir = os.path.join(output_dir,'labels','train')
    os.makedirs(output_labels_train_dir, exist_ok=True)
    export_coco_to_yolo(output_annotations_train_file, output_labels_train_dir)

    output_labels_val_dir = os.path.join(output_dir,'labels','val')
    os.makedirs(output_labels_val_dir, exist_ok=True)
    export_coco_to_yolo(output_annotations_val_file, output_labels_val_dir)

    output_labels_test_dir = os.path.join(output_dir,'labels','test')
    os.makedirs(output_labels_test_dir, exist_ok=True)
    export_coco_to_yolo(output_annotations_test_file, output_labels_test_dir)

    # Save yolo dataset yaml file
    cats = coco_dataset.get_categories()

    cat_str = {c['id']: c['name'] for c in cats}
    output_yolo_data_file = os.path.join(output_dir, 'yolo_dataset.yaml')
    yolo_dataset_cgf = {'path': output_dir,
                        'train': output_images_train_folder,
                        'val': output_images_val_folder,
                        'names': cat_str}

    with open(output_yolo_data_file, 'w') as f:
        yaml.dump(yolo_dataset_cgf, f, default_flow_style=False, sort_keys=False)

    # #----------------- make output dataset coco yaml file for ultralytics -------------------
    # cat = []
    # for i, c in enumerate(coco_dataset.categories):
    #     if c['id'] != i+1:
    #         raise Exception('unordered categories')
    #     cat.append(c['name'])
    #
    # data = {'path': output_dir,
    #         'train': 'images/train',
    #         'val': 'images/val',
    #         'test': 'images/test',
    #         'annotations': {'train': 'annotations/instances_train.json',
    #                          'val': 'annotations/instances_val.json',
    #                          'test': 'annotations/instances_test.json'},
    #         'names': cat
    #         }
    # with open(output_dataset_yolo_yaml_file, 'w') as f:
    #     yaml.dump(data, f, sort_keys=False)


def export_coco_to_yolo(coco_dataset_file, output_labels_dir):
    """
    export annotations from a coco dataset file to yolo dataset labels format

    :param coco_dataset_file - coco dataset file
    :param output_labels_dir - labels output dir
    """

    # Load COCO annotations
    if not os.path.isfile(coco_dataset_file):
        raise Exception('coco dataset json file: {}'.format(coco_dataset_file))
    coco_dataset = coco_dataset_utils.CocoDatasetManager()
    coco_dataset.load_coco(coco_dataset_file, verify_images = True)

    # Get category ID to name mapping (and reverse)
    cats = coco_dataset.get_categories()
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(cats)}
    cat_id_to_name = {cat['id']: cat['name'] for cat in cats}

    # Convert annotations
    img_ids = coco_dataset.get_image_ids()
    for img_id in img_ids:
        img_info = coco_dataset.get_image(image_id=img_id)
        file_name = img_info['file_name']
        width, height = img_info['width'], img_info['height']

        # Get annotations
        anns = coco_dataset.get_annotations(img_id)

        # Write YOLO labels
        label_path = os.path.join(output_labels_dir, os.path.splitext(os.path.basename(file_name))[0] + '.txt')
        label_lines = []
        if anns is not None:
            for ann in anns:
                if ann is not None:
                    # if ann.get('iscrowd', 0) == 1:
                    #     continue
                    x, y, w, h = ann['bbox']
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w /= width
                    h /= height
                    class_id = cat_id_to_index[ann['category_id']]
                    label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
    return



if __name__ == '__main__':
    baseline_dataset_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/coco_dataset.json'
    res_dataset_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco'

    # CONFIG
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    seed = 42
    random.seed(seed)

    # run
    print('loading baseline dataset: {}'.format(baseline_dataset_folder))
    baseline_dataset = coco_dataset_utils.CocoDatasetManager()
    baseline_dataset.load_coco(baseline_dataset_folder)
    baseline_dataset.verify()

    print('exporting dataset to yolo format at: {}'.format(res_dataset_folder))
    export_for_training(baseline_dataset_folder, res_dataset_folder,
                        split_ratios=split_ratios, augment_crop_size=(320, 240), augment_crop_number=3)