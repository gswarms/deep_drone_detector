import copy
import os
import shutil
import random
import pathlib
import time
from distutils.dir_util import copy_tree
import json

import cv2
import numpy as np
import yaml
import coco_dataset_manager
import dataset_training_utils


# TODO: use new coco dataset manager
# TODO: balance by bbox size
# TODO: balance by annotted / non-annotated
# TODO: balance by images metadata
# TODO: balance by annotations metadata


class CocoToUltralyticsYoloExporter:
    def __init__(self, input_coco_dataset_json):
        """
        export coco dataset for yolo ultralytics training format.
        main functionality:
        1. split to train/val/test
        2. ultralytics yolo dataset training format
        3. change image size
        4. balance by categories / annotation metadata / images metadata

        output dataset structure:
        yolo_output_dir/
        ├── yolo_data.yaml
        ├── annotations/
        │   ├── instances_train.json
        │   ├── instances_val.json
        │   └── instances_test.json
        ├── images/
        │   ├── train
        │   │   ├── 000001.jpg
        │   │   ├── 000002.jpg
        │   │   ├── ...
        │   │   ├── <imgid n>.txt
        │   ├── val
        │   │   ├── 000<n+1>.jpg
        │   │   ├── ...
        │   │   ├── 000<n+m+1>.jpg
        │   ├── test
        │   │   ├── 00<n+m+2>.jpg
        │   │   ├── ...
        │   │   ├── 000<n+m+2+k>.jpg

        * image file name in json annotation files must be relative to images/{split}.
          so: "file_name": "000001.jpg" and not "file_name": "images/train/000001.jpg"


        while yolo_data.yaml format:

        path: /absolute/path/to/dataset

        train: images/train
        val: images/val
        test: images/test

        nc: 3
        names:
          0: person
          1: car
          2: airplane

        annotations:
          train: annotations/instances_train.json
          val: annotations/instances_val.json
          test: annotations/instances_test.json


        :param input_coco_dataset_json: json file for input coco dataset
        """

        # load dataset
        self.coco_dataset = coco_dataset_manager.CocoDatasetManager()
        self.coco_dataset.load_coco(input_coco_dataset_json, verify_image_files=True)
        print('loaded coco dataset from: {}'.format(input_coco_dataset_json))


    def export(self, output_dataset_root_folder, split_ratios={'train': 0.7, 'val': 0.15, 'test':0.15},
                            augment_crop=None):
        """
        export dataset to ultralytics yolo training format

        :param output_dataset_root_folder: result dataset output folder
        :param image_size: result dataset image size
                           None - keep original image sizes
        :param augment_crop: parameters for augment crop
                             {'image_size': (width, height), 'num_samples': n}
                             None - no augment-crop will be done.
        :return:
        """

        # handle inputs
        if augment_crop is not None:
            known_keys = {"image_size", "num_samples"}
            if not(isinstance(augment_crop, dict) and augment_crop.keys() <= known_keys):
                raise Exception('invalid augment_crop parameters')

        #------------------- init output folders and files ----------------------------
        # output image folders
        output_images_folder = pathlib.Path(output_dataset_root_folder) / 'images'
        output_images_folder.mkdir(parents=True, exist_ok=True)
        output_images_train_folder = output_images_folder / 'train'
        output_images_train_folder.mkdir(parents=True, exist_ok=True)
        output_images_val_folder = output_images_folder / 'val'
        output_images_val_folder.mkdir(parents=True, exist_ok=True)
        output_images_test_folder = output_images_folder / 'test'
        output_images_test_folder.mkdir(parents=True, exist_ok=True)
        # output annotation files
        # output_annotations_folder = pathlib.Path(output_dataset_root_folder) / 'annotations'
        # output_annotations_folder.mkdir(parents=True, exist_ok=True)
        # output_annotations_train_file = output_annotations_folder / 'instances_train.json'
        # output_annotations_val_file = output_annotations_folder / 'instances_val.json'
        # output_annotations_test_file = output_annotations_folder / 'instances_test.json'
        output_annotations_folder = pathlib.Path(output_dataset_root_folder) / 'labels'
        output_annotations_folder.mkdir(parents=True, exist_ok=True)
        output_annotations_train_folder = output_annotations_folder / 'train'
        output_annotations_train_folder.mkdir(parents=True, exist_ok=True)
        output_annotations_val_folder = output_annotations_folder / 'val'
        output_annotations_val_folder.mkdir(parents=True, exist_ok=True)
        output_annotations_test_folder = output_annotations_folder / 'test'
        output_annotations_test_folder.mkdir(parents=True, exist_ok=True)
        output_annotations_train_file = output_annotations_folder / 'instances_train.json'
        output_annotations_val_file = output_annotations_folder / 'instances_val.json'
        output_annotations_test_file = output_annotations_folder / 'instances_test.json'


        # output ultralytics-yolo dataset config file
        output_yolo_yaml_file = pathlib.Path(output_dataset_root_folder) / 'ultralytics_dataset_data.yaml'

        #------------------- init train/val/test output datasets ----------------------------
        # init train/val/test output datasets
        train_dataset = coco_dataset_manager.CocoDatasetManager()
        train_dataset.set_root(output_dataset_root_folder)
        val_dataset = coco_dataset_manager.CocoDatasetManager()
        val_dataset.set_root(output_dataset_root_folder)
        test_dataset = coco_dataset_manager.CocoDatasetManager()
        test_dataset.set_root(output_dataset_root_folder)
        # copy categories
        train_dataset.df_categories = copy.deepcopy(self.coco_dataset.df_categories)
        val_dataset.df_categories = copy.deepcopy(self.coco_dataset.df_categories)
        test_dataset.df_categories = copy.deepcopy(self.coco_dataset.df_categories)
        # categories = self.coco_dataset.get_categories()
        # for cat_id in categories:
        #     cat = categories[cat_id]
        #     cat_id1 = train_dataset.add_category(cat['name'], cat['supercategory'])
        #     assert cat_id1 == cat_id, 'category id missmatch!'
        #     cat_id2 = val_dataset.add_category(cat['name'], cat['supercategory'])
        #     assert cat_id2 == cat_id, 'category id missmatch!'
        #     cat_id3 = test_dataset.add_category(cat['name'], cat['supercategory'])
        #     assert cat_id3 == cat_id, 'category id missmatch!'

        #---------------------- resize / crop dataset -----------------------
        # this is specific to our application!


        #---------------------- balance dataset -----------------------


        #----------------- split to train / test / val sets -------------------
        print('splitting dataset to train / test / val sets')

        # get all images
        image_ids = self.coco_dataset.get_image_ids()

        # split dataset to train / val / test
        # 0=train, 1=val, 2=test
        n = len(image_ids)
        dataset_types = np.random.choice((0,1,2), size=n, p=(split_ratios['train'], split_ratios['val'], split_ratios['test']))

        data = [{'name': 'train', 'dataset':train_dataset, 'dataset_type': 0,
                 'annotations_file': output_annotations_train_file,
                 'annotations_folder': output_annotations_train_folder,
                 'images_folder': output_images_train_folder},
                {'name': 'val', 'dataset': val_dataset, 'dataset_type': 1,
                 'annotations_file': output_annotations_val_file,
                 'annotations_folder': output_annotations_val_folder,
                 'images_folder': output_images_val_folder},
                {'name': 'test', 'dataset': test_dataset, 'dataset_type': 2,
                 'annotations_file': output_annotations_test_file,
                 'annotations_folder': output_annotations_test_folder,
                 'images_folder': output_images_test_folder}]
        for d in data:
            print('   making {} dataset...'.format(d['name']))
            t0 = time.monotonic()
            img_count = 0
            ann_count = 0
            for i in np.where(dataset_types == d['dataset_type'])[0]:
                img_count += 1
                image_id = image_ids[i]
                # get image
                img_data = self.coco_dataset.get_image(image_id=image_id)
                img_size = (int(img_data['width']), int(img_data['height']))
                img_path = str(self.coco_dataset.images_folder / img_data['file_name'])
                # get annotations
                img_annotations_dict = self.coco_dataset.get_image_annotations(image_id)

                if augment_crop is None:
                    # add image
                    img_id_new = d['dataset'].add_image(img_path, img_size, metadata=img_data['metadata'])
                    # add annotations
                    for ann in img_annotations_dict:
                        d['dataset'].add_annotation(img_id_new, category_id=ann['category_id'], bbox=ann['bbox'],
                                                 area=ann['area'], iscrowd=ann['iscrowd'], metadata=ann['metadata'])
                        ann_count += 1
                else:
                    raise Exception('not implemented yet')
                    cnt = 0
                    img = cv2.imread(img_path)
                    for j in range(augment_crop["num_samples"]):
                        cnt += 1
                        ann_bboxes = [x['bbox'] for x in img_annotations_dict]
                        rescale_factor, crop_bbox = self._get_random_crop_resize(img_size, ann_bboxes, augment_crop["image_size"])
                        rescale, crop_bbox = self._get_randon_crop_resize(img_size)
                        xtl,ytl, w, h = crop_bbox
                        img_tmp = img[xtl:xtl+w, ytl:ytl+h]
                        if rescale_factor != 1:
                            img_tmp = cv2.resize(img_tmp, augment_crop["image_size"])
                        # save new image
                        new_img_path = img_path.split('.')[0] + '_rszcrp{}'.format(cnt) + img_path.split('.')[1]
                        cv2.imwrite(new_img_path, img_tmp)
                        # add image
                        img_id_new = d['dataset'].add_image(new_img_path, augment_crop["image_size"], metadata=img_data['metadata'])
                        # add annotations
                        for ann in img_annotations_dict:
                            # convert annotation bbox to cropped-resized image
                            if rescale:
                                wi = ann['bbox'][2] * augment_crop["image_size"][0] / w
                                hi= ann['bbox'][3] * augment_crop["image_size"][1] / h
                            else:
                                wi, hi = [w, h]
                            # add
                            ann_bbox = (ann['bbox'][0] - xtl, ann['bbox'][1] - ytl, wi, hi)
                            d['dataset'].add_annotation(img_id_new, category_id=ann['category_id'], bbox=ann_bbox,
                                                        area=ann['area'], iscrowd=ann['iscrowd'], metadata=ann['metadata'])

            print('      added {} images, {} annotations, {:.2f}[sec]'.format(img_count, ann_count, time.monotonic()-t0))

            # save
            t0 = time.monotonic()
            d['dataset'].save_coco(dataset_root_folder=None, json_file_name=pathlib.Path(d['annotations_file']).name,
                                    copy_images=True, overwrite = False)
            print('      saved {:.2f}[sec]'.format(time.monotonic()-t0))

            # move images to fit ultralytics format
            t0 = time.monotonic()
            count = 0
            for item in output_images_folder.iterdir():
                if item.is_file():  # only files, skip subfolders
                    shutil.move(str(item), d['images_folder'] / item.name)
                    count += 1
            print('      moved {} images, {:.2f}[sec]'.format(count, time.monotonic() - t0))
            assert count == d['dataset'].df_images.shape[0], 'missmatch in number of copied images!'

            # save ultralytics labels
            img_ids = d['dataset'].get_image_ids()
            for i in img_ids:
                img_data = d['dataset'].get_image(i)
                ann_data = d['dataset'].get_image_annotations(i)
                txt_file = str(pathlib.Path(d['annotations_folder']) / '{}.txt'.format(pathlib.Path(img_data['file_name']).stem))
                with open(txt_file, "w") as f:
                    for a in ann_data:
                        xc = (a['bbox'][0] + a['bbox'][2] / 2) / img_data['width']
                        yc = (a['bbox'][1] + a['bbox'][3] / 2) / img_data['height']
                        w = a['bbox'][2] / img_data['width']
                        h = a['bbox'][3] / img_data['height']
                        txt_line = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(a['category_id'], xc, yc, w, h)
                        # Write one line per annotation
                        f.write(txt_line)

        #----------------- Save yolo dataset yaml file -------------------
        print('saving yolo dataset yaml file at: {}'.format(output_yolo_yaml_file))
        self._save_ultralytics_yolo_config_file(output_yolo_yaml_file, output_dataset_root_folder)


    def _save_ultralytics_yolo_config_file(self, output_yolo_yaml_file, output_dataset_root_folder):
        cats = self.coco_dataset.get_categories()
        cat_str = {k: {'name': v['name']} for k, v in cats.items()}
        # ann_str = {'train': 'annotations/instances_train.json',
        #            'val': 'annotations/instances_val.json',
        #            'test': 'annotations/instances_test.json'}
        yolo_dataset_cgf = {'path': output_dataset_root_folder,
                            'train': 'images/train',
                            'val': 'images/val',
                            'test': 'images/test',
                            'nc': len(cats),
                            'names': cat_str}
                            # 'annotations': ann_str}
        with open(output_yolo_yaml_file, 'w') as f:
            yaml.dump(yolo_dataset_cgf, f, default_flow_style=False, sort_keys=False)


    def _get_random_crop_resize(self, full_img_size, annotation_bboxes, required_image_size):
        """
        resize dataset images to a fixed image size - for training

        strategy:
        1. calculate a minimal containing bbox for all annotations
          TODO: this is not generally good! replace this step so that we also get crops with part of the annotations

        2. crop / resize
            - if annotation bbox is smaller than the required image size:
              crop the image to the required size
                - random crop
                - make sure relevant annotations are completely inside the crop.
                - make sure no annotations are partially inside the crop.
                - no resize

            - if annotation bbox is larger than the required image size:
                - find scale factor so that annotation bbox fits into required image size
                - randomly choose scale to be 1-0.7 of the required scale to make the bbox a little smaller
                - randomly crop requires image size so that annotations bbox is fully inside

                TODO: what if annotations bbox is too large, we can't resize and padd?

        :param full_img_size: full image size (width, height)
        :param annotation_bboxes: annotation bboxes - list of [[xtl, ytl, w, h], ...]
        :param required_image_size: required image size (width, height)
        :return:
        """
        raise Exception('not implemented')
        if isinstance(annotation_bboxes, list):
            if not isinstance(annotation_bboxes[0], list):
                annotation_bboxes = [annotation_bboxes]
        else:
            raise Exception('invalid input')

        xtl = [x[0] for x in annotation_bboxes]
        ytl = [x[0] for x in annotation_bboxes]
        xbr = [x[0] + x[2] for x in annotation_bboxes]
        ybr = [x[1] + x[3] for x in annotation_bboxes]
        bbox = [min(xtl), min(ytl), max(xbr) - min(xtl), max(ybr) - min(ytl)]


        for bbox in annotation_bboxes:
            # choose ROI valid size as a factor of bbox size
            sx = np.random.rand() * 1 + 1
            sy = np.random.rand() * 1 + 1
            w = bbox[2] * sx
            h = bbox[3] * sy

            # fit to required image size scale

            # required_image_size minimum limit

            # random crop around bbox

        rescale_factor = None
        crop_bbox = None

        return rescale_factor, crop_bbox


def generate_system_mimic_crop(image, bboxes, target_idx=0, base_size=256):
    img_h, img_w = image.shape[:2]

    # 1. Get the actual target location (center)
    # YOLO format: [class, xc, yc, w, h] normalized
    target = bboxes[target_idx]
    true_xc, true_yc = target[1] * img_w, target[2] * img_h

    # 2. Simulate Estimator Error (Shift)
    # Adjust 'sigma' to match your real-world covariance
    sigma = 40
    est_xc = true_xc + np.random.normal(0, sigma)
    est_yc = true_yc + np.random.normal(0, sigma)

    # 3. Simulate Covariance/Uncertainty (Scale)
    # Sometimes we take a 256 crop, sometimes larger (up to 480)
    capture_size = int(np.random.uniform(base_size, 480))

    # 4. Calculate Crop Coordinates
    x1 = int(est_xc - capture_size / 2)
    y1 = int(est_yc - capture_size / 2)

    # Ensure the crop contains the target (Requirement #3 & #4)
    # If the simulated error pushed the target out, clip it
    tw, th = target[3] * img_w, target[4] * img_h
    x1 = max(min(x1, int(true_xc - tw / 2)), int(true_xc + tw / 2 - capture_size))
    y1 = max(min(y1, int(true_yc - th / 2)), int(true_yc + th / 2 - capture_size))

    # Clamp to image boundaries
    x1 = max(0, min(x1, img_w - capture_size))
    y1 = max(0, min(y1, img_h - capture_size))

    # 5. Execute Crop and Resize
    crop = image[y1:y1 + capture_size, x1:x1 + capture_size]
    final_img = cv2.resize(crop, (base_size, base_size))

    # 6. Update Labels
    # We must account for the shift (x1, y1) AND the scaling factor
    scale = base_size / capture_size
    new_bboxes = []
    for b in bboxes:
        bx_px = b[1] * img_w
        by_px = b[2] * img_h
        bw_px = b[3] * img_w
        bh_px = b[4] * img_h

        # New center relative to crop and then scaled
        nx = (bx_px - x1) * scale / base_size
        ny = (by_px - y1) * scale / base_size
        nw = (bw_px * scale) / base_size
        nh = (bh_px * scale) / base_size

        # Only keep if the object is still mostly in the frame
        if 0 < nx < 1 and 0 < ny < 1:
            new_bboxes.append([b[0], nx, ny, nw, nh])

    return final_img, new_bboxes


if __name__ == '__main__':

    # base_dataset_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211'
    # input_coco_dataset_json = os.path.join(base_dataset_folder, 'merged_dataset_raw/annotations/coco_dataset.json')
    # output_dataset_root_folder = os.path.join(base_dataset_folder, 'ultalytics_yolo_20260121')
    # split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    # # augment_crop = {'image_size': (256, 256), 'num_samples': 4}
    # augment_crop = None


    base_dataset_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20260323'
    input_coco_dataset_json = os.path.join(base_dataset_folder, 'merged_dataset_raw/annotations/coco_dataset.json')
    output_dataset_root_folder = os.path.join(base_dataset_folder, 'ultalytics_yolo_20260324')
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    # augment_crop = {'image_size': (256, 256), 'num_samples': 4}
    augment_crop = None

    seed = 42
    random.seed(seed)

    c2y_exporter = CocoToUltralyticsYoloExporter(input_coco_dataset_json)
    c2y_exporter.export(output_dataset_root_folder, split_ratios=split_ratios, augment_crop=None)

