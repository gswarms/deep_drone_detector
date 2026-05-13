import copy
import os
import shutil
import random
import pathlib
import time
import cv2
import numpy as np
import yaml
import coco_dataset_manager

# TODO: use new coco dataset manager
# TODO: balance by bbox size
# TODO: balance by annotted / non-annotated
# TODO: balance by images metadata
# TODO: balance by annotations metadata


class CocoDatasetRefinery:
    def __init__(self, input_coco_dataset_json, log_file=None, verbose=True):
        """
        refine dataset for NN model training:
        - balance background / annotated images
        - resize/crop all images
        - TODO: add more options

        :param input_coco_dataset_json: json file for input coco dataset
        :param log_file: path to log file
        :param verbose: print log to screen
        """

        # load dataset
        self.coco_dataset = coco_dataset_manager.CocoDatasetManager()
        self.coco_dataset.load_coco(input_coco_dataset_json, verify_image_files=True)

        # setup log
        self.verbose = verbose
        self.log = None
        if log_file is not None:
            if self.verbose:
                print('log to: {}'.format(log_file))
            self.log = open(log_file, "w")

        num_images = len(self.coco_dataset.df_images['id'])
        num_annotated_images = len(list(set(self.coco_dataset.df_annotations['image_id'])))  # sorted
        num_annotations = len(self.coco_dataset.df_annotations)
        num_categories = len(self.coco_dataset.df_categories)
        self._log('loaded dataset from: {}'.format(str(pathlib.Path(input_coco_dataset_json).parent)))
        self._log('   {} images'.format(num_images))
        self._log('   {} annotated images'.format(num_annotated_images))
        self._log('   {} annotations'.format(num_annotations))
        self._log('   {} classes'.format(num_categories))

    def balance_background(self, background_ratio: float, balance_method):
        """
        balance the number of background images in the dataset by adding / removing images
        * Important Note: adding/removing images will actually copy and delete image files in the images folder.

        :param background_ratio: ratio of background images in the dataset = num_background_images / num_total_images
        :param balance_method: 'dilute' - delete labled/bckgnd images to get to the desired ratio
                               'expand' - duplicate labled/bckgnd images to get to the desired ratio
        :param verbose: verbose mode
        :return:
        """

        # handle inputs
        if background_ratio < 0 or background_ratio > 0:
            raise Exception('background_ratio must be between 0 and 1')
        if balance_method not in ['dilute', 'expand']:
            raise Exception('balance_method must be `dilute` or `expand`')

        self._log('background balance:')
        self._log('   parameters:')
        self._log('   - background_ratio = {}'.format(background_ratio))
        self._log('   - balance_method = {}'.format(balance_method))

        # balance background
        image_ids = self.coco_dataset.df_images['id']  # unique
        image_ids_annotated = list(set(self.coco_dataset.df_annotations['image_id']))  # sorted
        image_ids_background = list(set([x for x in image_ids if x not in image_ids_annotated]))  # sorted

        num_images = len(image_ids)
        num_annotated_images = len(image_ids_annotated)
        num_background_images = len(image_ids_background)
        background_ratio_curr = float(num_background_images) / float(num_images)
        num_annotations = len(self.coco_dataset.df_annotations)
        self._log('   pre balance dataset status:')
        self._log('   - {} images'.format(num_images))
        self._log('   - {} background images'.format(num_background_images))
        self._log('   - {} annotated images'.format(num_annotated_images))
        self._log('   - {} annotations'.format(num_annotations))
        self._log('   - background ratio = {}'.format(background_ratio_curr))

        if background_ratio['method'] == 'dilute':
            if background_ratio_curr > background_ratio['ratio']:
                # dilute background images

                # 1. calc number of images to remove
                # na - number of annotated images (given)
                # x - required number of background images
                # solve:
                # x / (na+x) = r
                # gives:
                # x = r * na / (1-r)
                n_required_background_images = int(np.round(background_ratio['ratio'] * num_annotated_images / (1 - background_ratio['ratio'])))

                # 2. remove images
                n_remove_images = num_background_images - n_required_background_images
                remove_index = np.round(np.linspace(0, num_background_images - 1, n_remove_images)).astype(np.uint32)
                remove_image_ids = [image_ids_background[i] for i in remove_index]
                for img_id in remove_image_ids:
                    img_data = self.coco_dataset.get_image(img_id)
                    # remove image from dataset
                    self.coco_dataset.remove_image(img_id)
                    # delete image file
                    img_file = pathlib.Path(img_data["file_name"])
                    if img_file.exists():
                        img_file.unlink()
                    else:
                        raise Exception('unable to remove image file: {} not found!'.format(str(img_file)))

            else:
                # dilute annotated images
                # 1. calc number of images to remove
                # nb - number of background images (given)
                # x - required number of annotated images
                # solve:
                # x / (x+nb) = (1-r)
                # gives:
                # x = nb * (1-r)/r
                n_required_annotated_images = int(np.round(num_background_images * (1 - background_ratio['ratio']) / background_ratio['ratio'] ))

                # 2. remove images
                n_remove_images = num_annotated_images - n_required_annotated_images
                remove_index = np.round(np.linspace(0, num_annotated_images - 1, n_remove_images)).astype(np.uint32)
                remove_image_ids = [image_ids_annotated[i] for i in remove_index]
                for img_id in remove_image_ids:
                    img_data = self.coco_dataset.get_image(img_id)
                    # remove image from dataset
                    self.coco_dataset.remove_image(img_id)
                    # delete image file
                    img_file = pathlib.Path(img_data["file_name"])
                    if img_file.exists():
                        img_file.unlink()
                    else:
                        raise Exception('unable to remove image file: {} not found!'.format(str(img_file)))

        elif background_ratio['method'] == 'expand':
            if background_ratio_curr < background_ratio['ratio']:
                # expand background images
                # 1. calc number of images to duplicate
                # na - number of annotated images (given)
                # x - required number of background images
                # solve:
                # x / (na+x) = r
                # gives:
                # x = r * na / (1-r)
                n_required_background_images = int(np.round(background_ratio['ratio'] * num_annotated_images / (1 - background_ratio['ratio'])))

                # 2. duplicate images
                n_add_images = n_required_background_images - len(image_ids_background)
                add_index = np.round(np.linspace(0, len(image_ids_background) - 1, n_add_images))
                add_image_ids = [image_ids_background[i] for i in add_index]
                self.coco_dataset.duplicate_image(add_image_ids)

            else:
                # expand annotated images
                # 1. calc number of images to duplicate
                # nb - number of background images (given)
                # x - required number of annotated images
                # solve:
                # x / (x+nb) = (1-r)
                # gives:
                # x = nb * (1-r)/r
                n_required_annotated_images = int(np.round(num_background_images * (1 - background_ratio['ratio']) / background_ratio['ratio'] ))

                # 2. duplicate images
                n_add_images = n_required_annotated_images - len(image_ids_annotated)
                add_index = np.round(np.linspace(0, len(image_ids_annotated) - 1, n_add_images))
                add_image_ids = [image_ids_annotated[i] for i in add_index]
                self.coco_dataset.duplicate_image(add_image_ids)

        else:
            raise Exception('invalid background_ratio method!')

        self.coco_dataset.save_coco()

        image_ids = self.coco_dataset.df_images['id']  # unique
        image_ids_annotated = set(self.coco_dataset.df_annotations['image_id'])  # sorted
        image_ids_background = set([x for x in image_ids if x not in image_ids_annotated])  # sorted
        num_images = len(image_ids)
        num_annotated_images = len(image_ids_annotated)
        num_background_images = len(image_ids_background)
        num_annotations = len(self.coco_dataset.df_annotations)
        new_background_ratio = float(num_background_images) / float(num_images)
        self._log('   post balance dataset status:')
        self._log('   - {} images'.format(num_images))
        self._log('   - {} background images'.format(num_background_images))
        self._log('   - {} annotated images'.format(num_annotated_images))
        self._log('   - {} annotations'.format(num_annotations))
        self._log('   - background ratio = {}'.format(new_background_ratio))

    def images_resize_crop(self, required_image_size, num_resizes, remove_original_annotated_images,
                           search_roi_scale_ratio=1, search_roi_min_width=100, search_roi_uncertainty_scale=(1, 4),
                           background_crop_scale=(1.5, 2.5)):
        """
        resize and crop all images in the dataset
        used mainly for training specific image sized

        :param required_image_size: (width, height)
        :param num_resizes: number of times to randomly resize / crop every image
        :param remove_original_annotated_images: boolean
                                                - True - remove the original uncropped image
                                                - False - keep the original uncropped image
        :param search_roi_scale_ratio:       random crop/rescale param - see _get_random_crop_resize
        :param search_roi_min_width:         random crop/rescale param - see _get_random_crop_resize
        :param search_roi_uncertainty_scale: random crop/rescale param - see _get_random_crop_resize
        :param background_crop_scale: (min, max) scale range for randomly cropping background images
                                      * original background image will always be saved as well
        :return:
        """

        self._log('image resize-crop:')
        self._log('   parameters:')
        self._log('   - required_image_size = {}'.format(required_image_size))
        self._log('   - num_resizes = {}'.format(num_resizes))
        self._log('   - remove_original_annotated_images = {}'.format(remove_original_annotated_images))
        self._log('   - search_roi_scale_ratio = {}'.format(search_roi_scale_ratio))
        self._log('   - search_roi_min_width = {}'.format(search_roi_min_width))
        self._log('   - search_roi_uncertainty_scale = {}'.format(search_roi_uncertainty_scale))
        self._log('   - background_crop_scale = {}'.format(background_crop_scale))

        image_ids = self.coco_dataset.df_images['id']  # unique
        for img_id in image_ids:
            img_data = self.coco_dataset.get_image(img_id)
            ann_data = self.coco_dataset.get_image_annotations(img_id)

            # get image
            img_file_name = (self.coco_dataset.images_folder / pathlib.Path(img_data['file_name'])).resolve()
            img = cv2.imread(img_file_name)
            full_img_size = (img.shape[1], img.shape[0])

            if len(ann_data) > 0:
                if len(ann_data) > 1:
                    raise Exception('more than one annotated image is not supported!')

                # crop/scale
                ann = ann_data[0]
                for i in range(num_resizes):
                    crop_bbox, rescale = _get_random_crop_resize(full_img_size, ann['bbox'], required_image_size,
                                                                 search_roi_scale_ratio=search_roi_scale_ratio,
                                                                 search_roi_min_width=search_roi_min_width,
                                                                 search_roi_uncertainty_scale=search_roi_uncertainty_scale)
                    img_crop = _img_crop_resize(img, crop_bbox, required_image_size)

                    # adjust annotation
                    ann_crop = [ann['bbox'][0] - crop_bbox[0], ann['bbox'][1] - crop_bbox[1], ann['bbox'][2], ann['bbox'][3]]
                    ann_crop = [int(np.floor(ann_crop[0] * rescale)),
                                int(np.floor(ann_crop[1] * rescale)),
                                int(np.ceil(ann_crop[2] * rescale)),
                                int(np.ceil(ann_crop[3] * rescale))]

                    # write image
                    new_img_file_name = img_file_name.with_name(img_file_name.stem + "_cs{:3d}".format(i) + img_file_name.suffix)
                    cv2.imwrite(new_img_file_name, img_crop)

                    # add to dataset
                    new_img_id = self.coco_dataset.add_image(str(new_img_file_name),
                                                             (img_crop.shape[1], img_crop.shape[0]),
                                                             img_data['metadata'])
                    self.coco_dataset.add_annotation(new_img_id, ann['category_id'], ann_crop, ann['segmentation'],
                                        ann_crop[2] * ann_crop[3], ann['iscrowd'], ann['metadata'])

                if remove_original_annotated_images:
                    raise Exception('remove_original_annotated_images not implemented yet!')

            else:  # this means it's a background image

                for i in range(num_resizes):
                    # crop / rescale image
                    if i==0:
                        pass # keep existing image
                    else:
                        w_min = int(np.round(full_img_size[0] / background_crop_scale[1]))
                        h_min = int(np.round(full_img_size[1] / background_crop_scale[1]))
                        w_max = int(np.round(full_img_size[0] / background_crop_scale[0]))
                        h_max = int(np.round(full_img_size[1] / background_crop_scale[0]))
                        w = np.random.randint(w_min, w_max)
                        h = np.random.randint(h_min, h_max)
                        xtl = np.random.randint(0, full_img_size[0] - w)
                        ytl = np.random.randint(0, full_img_size[1] - h)
                        crop_bbox = [xtl, ytl, w, h]
                        img_crop = _img_crop_resize(img, crop_bbox, required_image_size)

                        # write image
                        new_img_file_name = img_file_name.with_name(img_file_name.stem + "_cs{:3d}".format(i) + img_file_name.suffix)
                        cv2.imwrite(new_img_file_name, img_crop)

                        # add to dataset
                        new_img_id = self.coco_dataset.add_image(str(new_img_file_name),
                                                                 (img_crop.shape[1], img_crop.shape[0]),
                                                                 img_data['metadata'])

        self.coco_dataset.save_coco()

        image_ids = self.coco_dataset.df_images['id']  # unique
        image_ids_annotated = set(self.coco_dataset.df_annotations['image_id'])  # sorted
        num_images = len(image_ids)
        num_annotated_images = len(image_ids_annotated)
        num_annotations = len(self.coco_dataset.df_annotations)
        self._log('   post transform dataset status:')
        self._log('   {} images'.format(num_images))
        self._log('   {} annotated images'.format(num_annotated_images))
        self._log('   {} annotations'.format(num_annotations))

    def _log(self, log_str):
        if self.log is not None:
            self.log.write(log_str + '\n')
        if self.verbose:
            print(log_str)

    def __del__(self):
        if self.log is not None:
            self.log.close()

def _get_random_crop_resize(full_img_size, annotation_bbox, required_image_size, search_roi_scale_ratio=1, search_roi_min_width=100, search_roi_uncertainty_scale=(1, 2)):
    """
    resize dataset images to a fixed image size - for training
    miminc operational system ROI crop/resize from object prior los

    assumptions:
    TODO: these constraints are simple and generic. consider using more project-specific constraints.
    - we get an estimated LOS to the object along with it's uncertainty.
      This translates to a "search ROI":
      1. the search ROI always contains the object bbox - assume the estimated los and uncertainty are consistent.
      2. there is a minimal size for the search ROI - simulates constant orientation error.
      3. search ROI size grows with the real object bbox size - simulates position error that translates to increasing
                                                         orientation error as range decreases.
      4. search ROI aspect ration is 1 - uniform uncertainty.
      5. uniform probability for object bbox location inside the search ROI.  TODO: probabilistically this is wrong (inverse)

    strategy:
    TODO: this is good only for one annotation per image! expand this for handling several annotations
    1. select search ROI bbox:
        a. select the search ROI width:
            - no less than minimum size
            - inflate the annotation bbox width in a random scale (by the scale range)
        b. search ROI height corresponds to roi_scale_ratio
        c. shrink search ROI to full frame size if needed
        d. shift ROI with a random offset, keeping the annotation inside, and keeping inside the image.
    2. crop/rescale window
        - if search ROI is smaller than the required image size:
            - crop window centered around the ROI (also considering full image size)
            - no resize.
        - if search ROI is bigger than the required image size:
            - crop window tightly fits search ROI (might be different aspect ratio)
            - rescale required

    :param full_img_size: full image size (width, height)
    :param annotation_bbox: annotation bbox - list of [xtl, ytl, w, h]
    :param required_image_size: required image size (width, height)
    :param search_roi_scale_ratio: scale ratio for ROI
    :param search_roi_min_width: minimal roi width (roi_min_height = width / roi_scale_ration)
    :param search_roi_uncertainty_scale: (scale_min, scale_max) the scale range for inflating the annotation bbox
    :return: crop_bbox
    :return: crop_rescale
    """

    annotation_bbox = np.array(annotation_bbox).flatten()
    if annotation_bbox.size != 4:
        raise Exception('annotation_bbox size is wrong!')

    if annotation_bbox[0] < 0 or annotation_bbox[0] + annotation_bbox[2] > full_img_size[0]-1 \
       or annotation_bbox[1] < 0 or annotation_bbox[1] + annotation_bbox[3] > full_img_size[1]-1:
        raise Exception('annotation_bbox[0] is outside full image size!')
    xtl_annotation = annotation_bbox[0]
    ytl_annotation = annotation_bbox[1]
    w_annotation = annotation_bbox[2]
    h_annotation = annotation_bbox[3]

    # handling the case where required_image_size > full_img_size
    # 1. reduce required_image_size to full_img_size while keeping the aspect ratio
    # 2. add this scale for inflating later
    if required_image_size[0] > full_img_size[1] or required_image_size[1] > full_img_size[1]:
        tmp_rescale = max(required_image_size[0] / full_img_size[1], required_image_size[1] / full_img_size[1])
        required_image_size_reduced = [int(np.round(required_image_size[0] / tmp_rescale)),
                                   int(np.round(required_image_size[1] / tmp_rescale))]
        required_to_full_rescale = required_image_size[0] / required_image_size_reduced[0]
    else:
        required_to_full_rescale = 1
        required_image_size_reduced = required_image_size

    # randomly select search ROI size
    search_roi_uncertainty_scale_res = search_roi_uncertainty_scale[0] + np.random.rand(1) * (search_roi_uncertainty_scale[1] - search_roi_uncertainty_scale[0])
    search_roi_w = max(annotation_bbox[2] * search_roi_uncertainty_scale_res, search_roi_min_width)
    search_roi_h = search_roi_w / search_roi_scale_ratio

    # shrink search ROI to fit the full image if needed
    search_roi_w, search_roi_h = _bbox_shrink_to_fit(search_roi_w, search_roi_h, full_img_size)

    # make sure search ROI is not smaller than the annotation
    search_roi_w = max(search_roi_w, w_annotation)
    search_roi_h = max(search_roi_h, h_annotation)

    # randomly select ROI position
    x_max = min(xtl_annotation, full_img_size[0] - search_roi_w)
    x_min = max(xtl_annotation + w_annotation - search_roi_w, 0)
    y_max = min(ytl_annotation, full_img_size[1] - search_roi_h)
    y_min = max(ytl_annotation + h_annotation - search_roi_h, 0)
    search_roi_xtl = np.random.randint(x_min, x_max + 1)
    search_roi_ytl = np.random.randint(y_min, y_max + 1)

    # get crop window using required_image_size
    scale_x = search_roi_w / required_image_size_reduced[0]
    scale_y = search_roi_h / required_image_size_reduced[1]
    scale_resize = max(scale_x, scale_y)
    if scale_resize <= 1:
        # crop window centered around the search ROI
        # no rescale
        crop_window_w = required_image_size_reduced[0]
        crop_window_h = required_image_size_reduced[1]
        rescale = 1

    else:
        # crop window tight around the search ROI
        # rescale needed
        if scale_x >= scale_y:
            crop_window_w = search_roi_w
            crop_window_h = int(np.round(crop_window_w * required_image_size_reduced[1] / required_image_size_reduced[0]))
        else:
            crop_window_h = search_roi_h
            crop_window_w = int(np.round(crop_window_h * required_image_size_reduced[0] / required_image_size_reduced[1]))
        # make sure it's inside the image
        crop_window_w, crop_window_h = _bbox_shrink_to_fit(crop_window_w, crop_window_h, full_img_size)
        rescale = required_image_size_reduced[0] / crop_window_w

    cx = search_roi_xtl + search_roi_w / 2
    cy = search_roi_ytl + search_roi_h / 2
    crop_window_xtl = int(np.round(cx - crop_window_w / 2))
    crop_window_ytl = int(np.round(cy - crop_window_h / 2))
    crop_window_xtl = min(max(0, crop_window_xtl), full_img_size[0] - crop_window_w)
    crop_window_ytl = min(max(0, crop_window_ytl), full_img_size[1] - crop_window_h)

    crop_bbox = [crop_window_xtl, crop_window_ytl, crop_window_w, crop_window_h]
    rescale = rescale * required_to_full_rescale

    assert (crop_bbox[0] >= 0 and crop_bbox[0] + crop_bbox[2] - 1 <= full_img_size[0] - 1)
    assert (crop_bbox[1] >= 0 and crop_bbox[1] + crop_bbox[3] - 1 <= full_img_size[1] - 1)

    return crop_bbox, rescale


def _img_crop_resize(img, crop_bbox, required_image_size):

    if len(img.shape) == 3:
        img_crop = img[crop_bbox[1]: crop_bbox[1] + crop_bbox[3],
        crop_bbox[0]: crop_bbox[0] + crop_bbox[2], :]
    elif len(img.shape) == 2:
        img_crop = img[crop_bbox[1]: crop_bbox[1] + crop_bbox[3],
        crop_bbox[0]: crop_bbox[0] + crop_bbox[2]]
    else:
        raise Exception('invalid image shape!')
    if crop_bbox[2] != required_image_size[0] or crop_bbox[3] != required_image_size[1]:
        img_crop = cv2.resize(img_crop, required_image_size)
    return img_crop


def _bbox_shrink_to_fit(bbox_w, bbox_h, image_size):
    """
    shrink bbox to fit a reference image size
    keep aspect ratio
    :param bbox: [xtl, ytl, w, h]
    :param image_size:  [width, height]
    :return: bbox_shrinked
    """

    # shrink search ROI to fit the full image if needed
    scale_x = bbox_w / image_size[0]
    scale_y = bbox_h / image_size[1]
    scale_resize = max(scale_x, scale_y)
    if scale_resize > 1:
        bbox_w_shrink = int(np.floor(bbox_w / scale_resize))
        bbox_h_shrink = int(np.floor(bbox_h / scale_resize))
    else:
        bbox_w_shrink = int(np.floor(bbox_w))
        bbox_h_shrink = int(np.floor(bbox_h))

    return bbox_w_shrink, bbox_h_shrink


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


    def export(self, output_dataset_root_folder, split_ratios={'train': 0.7, 'val': 0.15, 'test':0.15}):
        """
        export dataset to ultralytics yolo training format

        :param output_dataset_root_folder: result dataset output folder
        :param image_size: result dataset image size
                           None - keep original image sizes
        :param augment_crop: parameters for augment crop
                             {'image_size': (width, height), 'num_samples': n}
                             None - no augment-crop will be done.
        :param background_balance: ratio of background images in the dataset
                                 None - keep all existing images
                                 method: 'dilute' - drop labled/bckgnd images to get to the desired ratio
                                         'expand' - duplicate labled/bckgnd images to get to the desired ratio
        :return:
        """

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

                # add image
                img_id_new = d['dataset'].add_image(img_path, img_size, metadata=img_data['metadata'])
                # add annotations
                for ann in img_annotations_dict:
                    d['dataset'].add_annotation(img_id_new, category_id=ann['category_id'], bbox=ann['bbox'],
                                             area=ann['area'], iscrowd=ann['iscrowd'], metadata=ann['metadata'])
                    ann_count += 1
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

        # ----------------- Save yolo dataset yaml file -------------------
        image_ids = self.coco_dataset.df_images['id']  # unique
        image_ids_annotated = set(self.coco_dataset.df_annotations['image_id'])  # sorted
        num_images = len(image_ids)
        num_annotated_images = len(image_ids_annotated)
        num_annotations = len(self.coco_dataset.df_annotations)

        dataset_size = {'images': num_images,
                        'annotated_images': num_annotated_images,
                        'annotations': num_annotations}
        export_params = {'split_params': split_ratios}

        output_log_file = os.path.join(output_dataset_root_folder, 'dataset_export_log.yaml')
        print('saving dataset export parameters at: {}'.format(output_log_file))
        log_data = {'dataset_source_folder': str(self.coco_dataset.root_folder),
                    'output_folder': output_dataset_root_folder,
                    'dataset_size': dataset_size,
                    'export_params': export_params}
        self._save_dataset_export_log(output_log_file, log_data)

        return


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

    @ staticmethod
    def _save_dataset_export_log(output_log_file, log_data):
        with open(output_log_file, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False)

if __name__ == '__main__':

    # base_dataset_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211'
    # input_coco_dataset_json = os.path.join(base_dataset_folder, 'merged_dataset_raw/annotations/coco_dataset.json')
    # output_dataset_root_folder = os.path.join(base_dataset_folder, 'ultalytics_yolo_20260121')
    # split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    # # augment_crop = {'image_size': (256, 256), 'num_samples': 4}
    # augment_crop = None
    # background_balance = {'ratio': None, 'method': None}

    # base_dataset_folder = '/home/roee/Projects/datasets/interceptor_drone/uav_detection_dataset/dataset_20260330'
    # input_coco_dataset_json = os.path.join(base_dataset_folder, 'merged_dataset_raw/annotations/coco_dataset.json')
    # output_dataset_root_folder = os.path.join(base_dataset_folder, 'ultalytics_yolo_20260330_bg_balanced')
    # split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    # # augment_crop = {'image_size': (256, 256), 'num_samples': 4}
    # augment_crop = None
    # background_balance = {'ratio': 0.15, 'method': 'dilute'}

    base_dataset_folder = '/home/roee/Projects/datasets/interceptor_drone/uav_detection_dataset/dataset_20260429'
    input_coco_dataset_json = os.path.join(base_dataset_folder, 'merged_dataset_raw/annotations/coco_dataset.json')
    output_dataset_root_folder = os.path.join(base_dataset_folder, 'ultalytics_yolo_20260429_bg_balanced')
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

    seed = 42
    random.seed(seed)

    # ----------------- refine dataset ---------------------------
    refinery = CocoDatasetRefinery(input_coco_dataset_json)

    # we have too many background images.
    # balance background ratio to 0.15 by diluting background images
    refinery.balance_background(background_ratio=0.15, balance_method='dilute')

    # resize images to 256x256 by random cropping
    refinery.images_resize_crop(required_image_size=(256, 256),
                                num_resizes=5,
                                remove_original_annotated_images=False,
                                search_roi_scale_ratio=1,
                                search_roi_min_width=50,
                                search_roi_uncertainty_scale=(1, 5),
                                background_crop_scale=(1.5, 2.5))

    # ---------------- export to ultralytics YOLO format ---------------------
    # load dataset
    raw_dataset = coco_dataset_manager.CocoDatasetManager()
    raw_dataset.load_coco(input_coco_dataset_json, verify_image_files=False)

    # export dataset
    c2y_exporter = CocoToUltralyticsYoloExporter(input_coco_dataset_json)
    c2y_exporter.export(output_dataset_root_folder, split_ratios=split_ratios)
