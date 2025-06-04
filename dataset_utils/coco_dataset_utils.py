import os
import json
import yaml
import cv2
from pathlib import Path
from typing import List, Dict


class CocoDatasetManager:
    def __init__(self):
        # image records
        self.image_records = []  # list of {'id': image id,
                                 #          'file_name': image absolute file path,
                                 #          'width': image width,
                                 #          'height': image height
                                 #          'scenario': scenario name      **optional
                                 #          'record_time': yyyymmdd_HHMMSS **optional
                                 #          'weather': weather description **optional
                                 #          }
        self.image_records_id_index = {}  # dict image id to data (only pointers, no data duplication)
        self.image_id_counter = 0
        self.curr_image_id = 1

        # image annotations
        self.annotations = []   # list of {'id': category id,
                                #          'image_id': image id,
                                #          'category_id': category id,
                                #          'bbox': [xtl, ytl, w, h],
                                #          'area': bbox area,
                                #          'iscrowd': flag that tells if this bbox is a single object or a crowd or
                                #                     an inseparable multiple objects
                                #                     ** optional
                                #           }
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
            new_id = self.add_image(img_file_name, image_width, image_height)
            image_id_map[img['id']] = new_id

        # add annotations
        for ann in coco['annotations']:
            new_ann = ann.copy()
            if 'iscrowd' in ann.keys():
                iscrowd = ann['iscrowd']
            else:
                iscrowd = 0
            self.add_annotation(image_id_map[new_ann['image_id']], category_id_map[new_ann['category_id']],
                                new_ann['bbox'], iscrowd=iscrowd)

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

        coco = {
            'images': coco_images,
            'annotations': self.annotations,
            'categories': self.categories
            }

        with open(out_json_path, 'w') as f:
            json.dump(coco, f, indent=2)

        print(f"COCO dataset exported to: {out_json_path}")


    def add_image(self, file_name, image_width, image_height):
        """
        add image to dataset

        :param file_name: image file name
        :param image_size: [width, height]
        :return: image dataset new id
        """
        new_id = self.image_id_counter + 1
        self.image_records.append({
            'id': new_id,
            'file_name': file_name,
            'width': image_width,
            'height': image_height
        })
        self.image_id_counter += 1
        self.image_records_id_index[new_id] = self.image_records[-1]

        return new_id


    def add_annotation(self, image_id, category_id, bbox, iscrowd=0):
        """
        add annotation to dataset

        :param image_id: corresponding image id
        :param category_id: category id
        :param bbox: [xtl, ytl, w, h]
        :param iscrowd: flag that tells if this bbox is a single object or a crowd or an inseparable multiple objects
        :return: annotation new id
        """
        annotation_id = self.annotation_id_counter + 1
        self.annotations.append({
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'area': bbox[2] * bbox[3],
            'iscrowd': iscrowd
        })
        self.annotation_id_counter += 1

        # update annotations_image_id_index
        if image_id not in self.annotations_image_id_index:
            self.annotations_image_id_index[image_id] = []
        self.annotations_image_id_index[image_id].append(self.annotations[-1])

        return annotation_id


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
            if supercategory is not 'none' and supercategory != super_category_existing:
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

        return self.annotations_image_id_index[image_id]


    def get_category_name_by_id(self, cat_id):
        res = None
        for k in self.categories:
            if self.categories[k]['id'] == cat_id:
                res = self.categories[k]['name']
        return res


    def get_category_id_by_name(self, cat_name):
        if cat_name in self.category_name_index:
            res = self.category_name_index[cat_name]['id']
        else:
            res = None
        return res


    def get_image(self, image_id=None):
        """
        get image record data

        :param image_id: image id to get
                         if None - get next image!
        :return: image data dict:
                     {'id': image id,
                     'file_name': image absolute file path,
                     'width': image width,
                     'height': image height
                     'scenario': scenario name      **optional
                     'record_time': yyyymmdd_HHMMSS **optional
                     'weather': weather description **optional
                     }
        """

        if image_id is None:
            image_id = self.curr_image_id

        if image_id in self.image_records_id_index:
            res_img_data = self.image_records_id_index[image_id]
        else:
            res_img_data = None
        self.curr_image_id = image_id + 1

        return res_img_data


    def _sort_images(self):
        """
        sort images by image file path
        useful if we want to later get images in order
        """
        self.image_records.sort(key=lambda x: x['file_name'])
        # fix image record index
        self.image_records_id_index = {x['id']: x for x in self.image_records}
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