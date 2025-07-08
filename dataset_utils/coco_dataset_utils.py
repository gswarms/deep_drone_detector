import copy
import os
import json
import yaml
import cv2
from pathlib import Path
from typing import List, Dict
from enum import Enum

from charset_normalizer.md import annotations


class ImageRecord:
    def __init__(self, id=None, file_name=None, width=None, height=None,
                 scenario=None, date=None, daytime=None, weather=None, cloud_coverage=None):
        """
        image record handler

        :param file_name: image id (int)
        :param file_name: image absolute file path
        :param width': image width (uint)
        :param height': image height (uint)
        :param scenario': scenario name (str) **optional
        :param date': yyyymmdd (str) **optional
        :param daytime': HHMMSS (str) **optional
        :param weather': weather description **optional
        :param cloud_coverage': cloud coverage [0,1] float **optional
        :return:
        """
        self.id = id
        self.file_name = file_name
        self.width = width
        self.height = height
        self.scenario = scenario
        self.date = date
        self.daytime = daytime
        self.weather = weather
        self.cloud_coverage = cloud_coverage
        return

    def from_dict(self, data_dict):
        """
        load from dict
        :param data_dict:
        :return:
        """

        dd = copy.deepcopy(data_dict)

        self.id = None
        self.file_name = None
        self.width = None
        self.height = None
        self.scenario = None
        self.date = None
        self.daytime = None
        self.weather = None
        self.cloud_coverage = None

        if 'id' in dd:
            self.id = dd['id']
        if 'file_name' in dd:
            self.file_name = dd['file_name']
        if 'width' in dd:
            self.width = dd['width']
        if 'height' in dd:
            self.height = dd['height']
        if 'scenario' in dd:
            self.scenario = dd['scenario']
        if 'date' in dd:
            self.date = dd['date']
        if 'daytime' in dd:
            self.daytime = dd['daytime']
        if 'weather' in dd:
            self.weather = dd['weather']
        if 'cloud_coverage' in dd:
            self.cloud_coverage = dd['cloud_coverage']


    def to_dict(self):
        """
        convert to dict
        Use only fields that are not None
        :param dd:
        :return:
        """

        res_dict = {}

        if self.id is not None:
            res_dict['id'] = self.id
        if self.file_name is not None:
            res_dict['file_name'] = self.file_name
        if self.width is not None:
            res_dict['width'] = self.width
        if self.height is not None:
            res_dict['height'] = self.height
        if self.scenario is not None:
            res_dict['scenario'] = self.scenario
        if self.date is not None:
            res_dict['date'] = self.date
        if self.daytime is not None:
            res_dict['daytime'] = self.daytime
        if self.weather is not None:
            res_dict['weather'] = self.weather
        if self.cloud_coverage is not None:
            res_dict['cloud_coverage'] = self.cloud_coverage

        return res_dict


class Annotation:
    def __init__(self, image_id=None, category_id=None, bbox=None, area=None, is_crowd=False, distance_from_camera=None):
        """
        annotation record handler

        :param image_id: corresponding image id (int)
        :param category_id: category id (int)
        :param bbox: bounding box [xtl, ytl, w, h]
        :param area: annotation area in the image (float)
        :param is_crowd: flag for a single object or a crowd of inseparable multiple objects (bool)
        :param is_crowd: object distance from the camera (float)
        :return:
        """
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox
        if area is None and bbox is not None:
            area = bbox[2] * bbox[3]
        self.area = area
        self.is_crowd = is_crowd
        self.distance_from_camera = distance_from_camera
        return


    def from_dict(self, data_dict):
        """
        load from dict
        :param dd:
        :return:
        """

        self.image_id = None
        self.category_id = None
        self.bbox = None
        self.area = None
        self.is_crowd = None
        self.distance_from_camera = None

        dd = copy.deepcopy(data_dict)

        if 'image_id' in dd:
            self.image_id = dd['image_id']
        if 'category_id' in dd:
            self.category_id = dd['category_id']
        if 'bbox' in dd:
            self.bbox = dd['bbox']
        if 'area' in dd:
            self.area = dd['area']
        if 'is_crowd' in dd:
            self.is_crowd = dd['is_crowd']
        if 'distance_from_camera' in dd:
            self.distance_from_camera = dd['distance_from_camera']

    def to_dict(self):
        """
        convert to dict
        Use only fields that are not None
        :param data_dict:
        :return:
        """

        res_dict = {}

        if self.image_id is not None:
            res_dict['image_id'] = self.image_id
        if self.category_id is not None:
            res_dict['category_id'] = self.category_id
        if self.bbox is not None:
            res_dict['bbox'] = self.bbox
        if self.area is not None:
            res_dict['area'] = self.area
        if self.is_crowd is not None:
            res_dict['is_crowd'] = self.is_crowd
        if self.distance_from_camera is not None:
            res_dict['distance_from_camera'] = self.distance_from_camera

        return res_dict


class CocoDatasetManager:
    def __init__(self):
        # image records
        self.image_records = []  # list of {'id': image id,
                                 #          'file_name': image absolute file path,
                                 #          'width': image width,
                                 #          'height': image height
                                 #          'scenario': scenario name (str)  **optional
                                 #          'date': yyyymmdd (str)           **optional
                                 #          'daytime': HHMMSS (str)          **optional
                                 #          'weather': weather description   **optional
                                 #          'cloud_coverage': cloud coverage [0,1] float  **optional
                                 #          'place': place name (str)  **optional
                                 #          }

        self.image_records_id_index = {}  # dict image id to data (only pointers, no data duplication)
        self.image_records_id_to_idx = {}  # dict image id to index
        self.image_id_counter = 0
        self.curr_image_index = 0

        # image annotations
        self.annotations = []   # list of {'id': category id,
                                #          'image_id': image id,
                                #          'category_id': category id,
                                #          'bbox': [xtl, ytl, w, h],
                                #          'area': bbox area,
                                #          'iscrowd': flag that tells if this bbox is a single object or a crowd or
                                #                     an inseparable multiple objects
                                #                     ** optional
                                #          'range_from_camera': distance from camera (float )  **optional
                                #           }
        self.annotations_id_to_index = {}  # dict annotation id to index
        self.annotations_image_id_index = {}  # dict image id to a list of all corresponding data (only pointers, no data duplication)
        self.annotation_id_counter = 0

        # categories
        self.categories = []  # list of {'id': category id,
                              #          'name': category name,
                              #          'supercategory': supercategory name  ** optional
                              #           }
        self.category_id_counter = 0
        self.category_name_index = {}  # dict category name to data (only pointers, no data duplication)


    def load(self, json_path: str, verify_images=True):
        """
        load coco dataset

        :param json_path: coco dataset json file
        :param verify_images:
        :return:
        """

        if not os.path.isfile(json_path):
            raise Exception('coco json file not found at {}'.format(json_path))

        with open(json_path, 'r') as f:
            coco = json.load(f)

        # add categories
        categories = coco.get('categories', [])
        category_id_map = {}
        for ct in categories:
            category_id = self.add_category(ct['name'], ct['supercategory'])
            category_id_map[ct['id']] = category_id

        # add images
        image_id_map = {}
        dataset_dir = os.path.dirname(json_path)
        for img in coco['images']:
            img_file_name = os.path.abspath(os.path.join(dataset_dir, img['file_name']))
            image_width = img.get('width', 0)
            image_height = img.get('height', 0)
            date = img.get('date', 0)
            daytime = img.get('daytime', 0)
            weather = img.get('weather', 0)
            cloud_coverage = img.get('cloud_coverage', 0)
            new_id = self.add_image(img_file_name, image_width, image_height,
                                    scenario=date, date=date, daytime=daytime,
                                    weather=weather, cloud_coverage=cloud_coverage)
            image_id_map[img['id']] = new_id

        # add annotations
        for ann in coco['annotations']:
            new_ann = ann.copy()
            if 'iscrowd' in ann.keys():
                iscrowd = ann['iscrowd']
            else:
                iscrowd = 0
            distance_from_camera = img.get('distance_from_camera', 0)
            self.add_annotation(image_id_map[new_ann['image_id']], category_id_map[new_ann['category_id']],
                                new_ann['bbox'], iscrowd=iscrowd, distance_from_camera=distance_from_camera)

        #------------------ sort by image file name ------------------------------------
        self._sort_images()

        #------------------ verify images exist ------------------------------------
        if verify_images:
            missing_images = self._verify_images_exist()
            if len(missing_images) > 0:
                for mi in missing_images:
                    print('image file missing: {}'.format(mi))
                raise Exception('missing image files')


    def load_yolo(self, yolo_dir: str, verify_images=True):
        """
        load yolo dataset

        :param yolo_dir:
        :param verify_images:
        :return:
        """

        if not os.path.isdir(yolo_dir):
            raise Exception('yolo dir not found at {}'.format(yolo_dir))

        yolo_yaml_file = os.path.join(yolo_dir, 'data.yaml')
        if not os.path.isfile(yolo_yaml_file):
            raise Exception('yolo yaml file not found at {}'.format(yolo_yaml_file))

        with open(yolo_yaml_file, 'r') as f:
            data_cfg = yaml.safe_load(f)
        category_names = data_cfg['names']

        images_dir = os.path.join(yolo_dir, 'images')
        labels_dir = os.path.join(yolo_dir, 'labels')
        if not os.path.isdir(images_dir):
            raise Exception('yolo images dir not found at {}'.format(images_dir))
        if not os.path.isdir(labels_dir):
            raise Exception('yolo labels dir not found at {}'.format(labels_dir))

        # add categories
        category_id_remap_index = {}
        for i, cn in enumerate(category_names):
            category_id = self.add_category(cn, 'none')
            category_id_remap_index[i] = category_id

        # add images and labels
        image_files = list(Path(images_dir).glob("*.jpg"))
        for image_file in image_files:

            # add image
            # TODO: get image size without reading it
            img = cv2.imread(str(image_file))
            if img is None:
                continue
            height, width = img.shape[:2]
            image_id = self.add_image(str(image_file.resolve()), width, height)

            # add labels
            label_file = Path(labels_dir) / image_file.name.replace(".jpg", ".txt")
            if os.path.isfile(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_id, x, y, w, h = map(float, parts)
                        remapped_category_id = category_id_remap_index[cls_id]
                        coco_bbox = self._yolo_to_coco_bbox(x, y, w, h, width, height)
                        self.add_annotation(image_id, remapped_category_id,
                                            coco_bbox, iscrowd=0)

        # sort images by file name
        self._sort_images()

        # verify images exist
        if verify_images:
            missing_images = self._verify_images_exist()
            if len(missing_images) > 0:
                for mi in missing_images:
                    print('image file missing: {}'.format(mi))
                raise Exception('missing image files')


    def save(self, out_json_path: str):
        export_dir = os.path.dirname(os.path.abspath(out_json_path))

        # convert image paths relative to coco dataset yaml file
        coco_images = []
        for img in self.image_records:
            rel_path = os.path.relpath(img['file_name'], start=export_dir)
            img_copy = img.copy()
            img_copy['file_name'] = rel_path
            coco_images.append(img_copy)

        # sort by image id (for readability)
        coco_images = sorted(coco_images, key=lambda x: x["id"])

        coco = {
            'images': coco_images,
            'annotations': self.annotations,
            'categories': self.categories
            }

        with open(out_json_path, 'w') as f:
            json.dump(coco, f, indent=2)

        print(f"COCO dataset exported to: {out_json_path}")


    def add_image(self, file_name, image_width, image_height, scenario=None, date=None, daytime=None, weather=None, cloud_coverage=None):
        """
        add image to dataset

        :param file_name: image file name
        :param image_size: [width, height]
        :param date: date image was taken yyyymmdd string  ** optional
        :param daytime: time of day HHMMSS string  ** optional
        :param cloud_coverage: scalar [0,1]  ** optional
        :param place: string the name of the place where this image was taken  ** optional
        :param scenario: string the name of the scenario
        :return: image dataset new id
        """
        new_id = self.image_id_counter + 1
        image_record = {
            'id': new_id,
            'file_name': os.path.abspath(file_name),
            'width': image_width,
            'height': image_height
        }
        if scenario is not None:
            image_record['scenario'] = scenario
        if date is not None:
            image_record['date'] = date
        if daytime is not None:
            image_record['daytime'] = daytime
        if weather is not None:
            image_record['weather'] = weather
        if cloud_coverage is not None:
            image_record['cloud_coverage'] = cloud_coverage

        self.image_records.append(image_record)
        self.image_id_counter += 1
        self.image_records_id_index[new_id] = self.image_records[-1]
        self.image_records_id_to_idx[new_id] = len(self.image_records) - 1

        return new_id

    def remove_image(self, image_id):
        """
        remove image from the dataset

        :param image_id: image id to remove
        """

        if image_id in self.image_records_id_to_idx:
            # find image index
            img_idx = self.image_records_id_to_idx[image_id]

            # remove image
            self.image_records.pop(img_idx)
            self.image_records_id_index.pop(image_id)

            # fix id->idx dict
            for imgid in self.image_records_id_to_idx:
                if self.image_records_id_to_idx[imgid] > img_idx:
                    self.image_records_id_to_idx[imgid] = self.image_records_id_to_idx[imgid] - 1
            self.image_records_id_to_idx.pop(image_id)

            # update self.curr_image_index
            if self.curr_image_index >= img_idx:
                self.curr_image_index = max(self.curr_image_index - 1, 0)

            # remove corresponding annotations
            ann = self.get_annotations(image_id)
            for a in ann:
                self.remove_annotation(a['id'])

            res = True

        else:
            res = False

        return res

    def add_annotation(self, image_id, category_id, bbox, iscrowd=0, distance_from_camera=None):
        """
        add annotation to dataset

        :param image_id: corresponding image id
        :param category_id: category id
        :param bbox: [xtl, ytl, w, h]
        :param iscrowd: flag that tells if this bbox is a single object or a crowd or an inseparable multiple objects
        :param distance_from_camera: distance of object from camera (scalar)  ** optional
        :return: annotation new id
        """
        annotation_id = self.annotation_id_counter + 1
        annotation = {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'area': bbox[2] * bbox[3],
            'iscrowd': iscrowd
        }
        if distance_from_camera is not None:
            annotation['distance_from_camera'] = distance_from_camera
        self.annotations.append(annotation)
        self.annotation_id_counter += 1
        self.annotations_id_to_index[annotation_id] = len(self.annotations) - 1

        # update annotations_image_id_index
        if image_id not in self.annotations_image_id_index:
            self.annotations_image_id_index[image_id] = []
        self.annotations_image_id_index[image_id].append(self.annotations[-1])

        return annotation_id

    def update_annotation(self, annotation_id, category_id, bbox, iscrowd=None, distance_from_camera=None):
        """
        update existing annotation

        :param annotation_id: existing annotation id
        :param category_id: category id
        :param bbox: [xtl, ytl, w, h]
        :param iscrowd: flag that tells if this bbox is a single object or a crowd or an inseparable multiple objects
        :param distance_from_camera: distance of object from camera (scalar)  ** optional
        :return: True/False - success flag
        """

        if annotation_id in self.annotations_id_to_index:
            idx = self.annotations_id_to_index[annotation_id]
            self.annotations[idx]["category_id"] = category_id
            self.annotations[idx]["bbox"] = bbox
            self.annotations[idx]["area"] = bbox[2] * bbox[3]
            if iscrowd is not None:
                self.annotations[idx]["iscrowd"] = iscrowd
            if distance_from_camera is not None:
                self.annotations[idx]['distance_from_camera'] = distance_from_camera
            res = True

        else:
            res = False

        return res

    def remove_annotation(self, annotation_id):
        """
        remove annotation from the dataset

        :param annotation_id: annotation id to remove
        """

        if annotation_id in self.annotations_id_to_index:
            # find annotation index
            ann_idx = self.annotations_id_to_index[annotation_id]

            # remove annotation
            self.annotations.pop(ann_idx)
            self.annotations_image_id_index.pop(annotation_id)

            # fix id->idx dict
            for annid in self.annotations_id_to_index:
                if self.annotations_id_to_index[annid] > ann_idx:
                    self.annotations_id_to_index[annid] = self.annotations_id_to_index[annid] - 1
            self.annotations_id_to_index.pop(annotation_id)

            # update self.curr_image_index
            if self.curr_image_index >= ann_idx:
                self.curr_image_index = max(self.curr_image_index - 1, 0)

            res = True

        else:
            res = False

        return res

    def add_category(self, category_name, supercategory='none'):
        """
        add category

        :param category_name: str
        :param supercategory: str
        :return: new category id in dataset
        """

        if category_name not in self.category_name_index:
            remapped_id = self.category_id_counter + 1
            self.categories.append({'id': remapped_id, 'name': category_name, 'supercategory': supercategory})
            self.category_name_index[category_name] = self.categories[-1]
            self.category_id_counter = self.category_id_counter + 1

        else:

            remapped_id =  self.category_name_index[category_name]['id']
            # check supercategory consistency
            super_category_existing = self.category_name_index[category_name]['supercategory']
            if supercategory != 'none' and supercategory != super_category_existing:
                raise Exception('category {} has supercategory inconsistency! {} vs {}'.format(category_name, supercategory, super_category_existing))

        return remapped_id


    def get_annotations(self, image_id):
        """
        get annotations corresponding to a specific image id

        :param image_id: image id
        :return: all annotations for a specific image. list of dicts:
                         {'id': category id,
                          'image_id': image id,
                          'category_id': category id,
                          'bbox': [xtl, ytl, w, h],
                          'area': bbox area,
                          'iscrowd': flag that tells if this bbox is a single object or a crowd or
                                     an inseparable multiple objects
                                     ** optional
                          }
        """
        if image_id in self.annotations_image_id_index:
            image_annotations = self.annotations_image_id_index[image_id]
        else:
            image_annotations = None
        return image_annotations


    def get_categories(self):
        """
        get categories

        :return: list of
                         {'id': category id,
                          'image_id': image id,
                          'category_id': category id,
                          'bbox': [xtl, ytl, w, h],
                          'area': bbox area,
                          'iscrowd': flag that tells if this bbox is a single object or a crowd or
                                     an inseparable multiple objects
                                     ** optional
                          }
        """
        return self.categories


    def get_category_name_by_id(self, cat_id):
        res = None
        for cat in self.categories:
            if cat['id'] == cat_id:
                res = cat['name']
        return res


    def get_category_id_by_name(self, cat_name):
        if cat_name in self.category_name_index:
            res = self.category_name_index[cat_name]['id']
        else:
            res = None
        return res


    def get_image(self, image_id=None, image_index=None):
        """
        get image record data by image id or by index

        :param image_id: image id to get
        :param image_index: image index to get
        * if both are None, we get the next image (by image index)

        :return: image data dict:
                     {'id': image id,
                     'file_name': image absolute file path,
                     'width': image width,
                     'height': image height
                     date: date image was taken yyyymmdd string  ** optional
                     daytime: time of day HHMMSS string          ** optional
                     cloud_coverage: scalar [0,1]                ** optional
                     place: string the name of the place where this image was taken  ** optional
                     scenario: string the name of the scenario                       **optional
                     'weather': weather description                                   **optional
                     }
        """
        if image_id is None and image_index is None:
            image_index = self.curr_image_index

        if image_id is None:
            res_img_data = self.image_records[image_index]
            self.curr_image_index = min(image_index + 1, len(self.image_records)-1)

        elif image_id in self.image_records_id_index:
            res_img_data = self.image_records_id_index[image_id]
            self.curr_image_index = self.image_records_id_index[image_id]['id'] + 1

        else:
            res_img_data = None

        return res_img_data

    def get_image_ids(self, only_annotated=False):
        """
        get all image ids
        :param only_annotated - bool flag.
                                False - get all image ids
                                True - get only annotated image ids
        :return:
        """
        if only_annotated:
            image_ids = [x for x in self.annotations_image_id_index.keys() if self.annotations_image_id_index[x] is not None and len(self.annotations_image_id_index[x])>0]
        else:
            image_ids = [x['id'] for x in self.image_records]

        return image_ids


    def verify(self):
        """
        verify coco data
        :return:
        """

        valid_image_counter = 0
        annotated_image_counter = 0
        for image_record in self.image_records:
            if os.path.isfile(image_record['file_name']):
                valid_image_counter = valid_image_counter + 1
            else:
                aa=5
            annotations = self.get_annotations(image_record['id'])
            if annotations is not None:
                annotated_image_counter = annotated_image_counter + 1

        print('found {} images:'.format(len(self.image_records)))
        print('   - {} valid image files'.format(valid_image_counter))
        print('   - {} annotated images'.format(annotated_image_counter))

        return


    def _sort_images(self):
        """
        sort images by image file path
        useful if we want to later get images in order
        """
        self.image_records.sort(key=lambda x: x['file_name'])
        # fix image record index
        self.image_records_id_index = {x['id']: x for x in self.image_records}
        self.image_records_id_to_idx = {x['id']: i for i,x in enumerate(self.image_records)}
        return


    @staticmethod
    def _yolo_to_coco_bbox(x, y, w, h, img_w, img_h):
        abs_x = x * img_w
        abs_y = y * img_h
        abs_w = w * img_w
        abs_h = h * img_h
        return [abs_x - abs_w / 2, abs_y - abs_h / 2, abs_w, abs_h]


    def _verify_images_exist(self) -> List[str]:
        """
        Checks whether all images referenced by `file_name` exist on disk.

        Returns:
            A list of missing image file paths.
        """
        missing = []
        for img in self.image_records:
            if not os.path.isfile(img['file_name']):
                missing.append(img['file_name'])
        if missing:
            print(f"Missing {len(missing)} images.")
        else:
            print("✅ All images verified.")
        return missing


if __name__ == '__main__':

    import glob

    # convert each yolo scenario to a coco scenario
    dataset_base_dir = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw'
    dataset_scearios = glob.glob(os.path.join(dataset_base_dir, '*', '*'))
    for scen_path in dataset_scearios:
        manager = CocoDatasetManager()
        manager.load_yolo(scen_path)
        coco_dataset_file = os.path.join(scen_path, 'coco_dataset.json')
        manager.save(coco_dataset_file)

    # join coco datasets to one dataset
    manager = CocoDatasetManager()
    dataset_base_dir = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw'
    dataset_scearios = glob.glob(os.path.join(dataset_base_dir, '*', '*'))
    for scen_path in dataset_scearios:
        coco_dataset_file = os.path.join(scen_path, 'coco_dataset.json')
        manager.load(coco_dataset_file)
    manager.save(os.path.join(dataset_base_dir, 'coco_dataset.json'))