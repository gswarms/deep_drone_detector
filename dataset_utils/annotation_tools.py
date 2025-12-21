import cv2
import os
from datetime import datetime
import numpy as np
import glob
from pathlib import Path
import argparse


class Annotator:
    def __init__(self, categories=['object'], window_name='Annotator'):
        """
        helper class for annotating images

        :param window_name: opencv window for draw folder
        """

        # current image variables
        self.current_image_idx = None
        self.current_annotation_boxes = []
        self.current_annotation_classes = []
        self.current_annotation_ids = []
        self.current_class = categories[0]
        self.current_image = None
        self.current_image_id = None

        # state variables
        self.drawing = False
        self.removing = False
        self.start_point = None

        self.categories = categories
        self.current_category_index = 0
        self.current_category = self.categories[self.current_category_index]
        print('current category is {}'.format(self.current_category))
        self.window_name = window_name

        self.model = None

    def save(self):
        raise NotImplementedError("Subclasses must implement save()")

    def load_image(self, idx_offset=1, color_space='BGR'):
        raise NotImplementedError("Subclasses must implement load_image()")

    def update_current_category(self):
        self.current_category_index = self.current_category_index + 1
        if self.current_category_index > len(self.categories) - 1:
            self.current_category_index = 0
        self.current_category = self.categories[self.current_category_index]
        print('current category is {}'.format(self.current_category))

    def update_current_image_annotations(self):
        """
        update current image annotation in dataset

        assume:
        self.current_annotation_boxes: all annotation bboxes
        self.current_annotation_classes: all annotation classes
        self.current_annotation_ids:
                for a new annotation, annotation id is None
                for an existing annotation, annotation id stays the same
        """
        raise NotImplementedError("Subclasses must implement update_current_image_annotations()")

    def clear_current_annotations(self):
        self.current_annotation_boxes = []
        self.current_annotation_classes = []

    def set_current_class(self, class_name):
        self.current_class = class_name

    def set_remove_state(self):
        self.removing = True

    def draw_current_annotations(self):
        annotated = self.current_image.copy()
        for (box, cls) in zip(self.current_annotation_boxes, self.current_annotation_classes):
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, cls, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return annotated

    @staticmethod
    def draw_annotations(img, boxes, classes):
        for (box, cls) in zip(boxes, classes):
            x1, y1, x2, y2 = box
            # x1, y1, w, h = box
            # x2 = x1 + w
            # y2 = y1 + h
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, cls, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return img

    def update_display(self, temp_box=None):
        """

        :param temp_box: new bbox to add to existing annotations. (xtl, ytl, xbr, ybr)
        :return:
        """
        img_copy = self.current_image.copy()
        temp_boxes = self.current_annotation_boxes.copy()
        temp_classes = self.current_annotation_classes.copy()
        # convert to xtl,ytl,xbr,ybr
        for i, tb in enumerate(temp_boxes):
            temp_boxes[i] = (tb[0], tb[1], tb[0]+tb[2], tb[1]+tb[3])

        # add new box
        if temp_box is not None:
            temp_boxes.append(temp_box)
            temp_classes.append(self.current_class)

        # round to int
        for i, tb in enumerate(temp_boxes):
            temp_boxes[i] = (int(np.round(tb[0])), int(np.round(tb[1])), int(np.round(tb[2])), int(np.round(tb[3])))
        drawn = self.draw_annotations(img_copy, temp_boxes, temp_classes)

        cv2.putText(drawn, "frame {}".format(self.current_image_id), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if self.removing:
            cv2.putText(drawn, "Click inside a box to REMOVE", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow(self.window_name, drawn)

    def mouse_callback(self, event, x, y, flags, param):
        """
        mouse callback

        :param event:
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        """
        if self.removing and event == cv2.EVENT_MBUTTONDOWN:
            box_removed = False
            for i, (x1, y1, x2, y2) in enumerate(self.current_annotation_boxes):
                if x1 <= x <= x1+x2 and y1 <= y <= y1+y2:
                    del self.current_annotation_boxes[i]
                    del self.current_annotation_classes[i]
                    print("Removed annotation.")
                    box_removed = True
                    break
            if not box_removed:
                print("No box removed. at ({},{})".format(x, y))
            self.removing = False
            self.update_display()
            return

        if event == cv2.EVENT_MBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.update_display(temp_box=(self.start_point[0], self.start_point[1], x, y))

        elif event == cv2.EVENT_MBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.start_point
            x1 = min(max(x1, 0), self.current_image.shape[1]-1)
            y1 = min(max(y1, 0), self.current_image.shape[0]-1)
            x2, y2 = x, y
            x2 = min(max(x2, 0), self.current_image.shape[1]-1)
            y2 = min(max(y2, 0), self.current_image.shape[0]-1)
            if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                self.current_annotation_ids.append(None)
                self.current_annotation_boxes.append((min(x1, x2), min(y1, y2), int(np.round(np.abs(x1 -x2))), int(np.round(np.abs(y1 - y2)))))
                self.current_annotation_classes.append(self.current_class)
            self.update_display()


class AnnotatorStandardRecordFolder(Annotator):
    # TODO: make sure record annotator works - seems invalid!
    def __init__(self, record_base_folder, categories, window_name='Annotator'):
        """
        helper class for annotating images

        :param image_folder: images folder
        :param annotations_file: annotations file
        """
        super().__init__(categories, window_name)  # Call base class __init__

        self.base_folder = record_base_folder
        if not os.path.isdir(record_base_folder):
            raise Exception('record base folder {} not found'.format(record_base_folder))

        self.images_folder = os.path.join(self.base_folder, 'images')
        if not os.path.isdir(self.images_folder):
            raise Exception('images folder {} not found'.format(self.images_folder))


        self.dataset_file = os.path.join(self.base_folder, 'coco_annotations.yaml')
        import coco_dataset_tmp as cdu
        self.dataset = cdu.CocoDatasetManager()
        if os.path.isfile(self.dataset_file):
            self.dataset.load_coco(coco_dataset_file)

        self.current_image_id = None

    def load(self, annotations_file):
        raise Exception('load not implemented yet!')
        # self.dataset.save(annotations_file)

    def save(self):
        raise Exception('save not implemented yet!')
        # self.dataset.save(annotations_file)

    def update_current_image_annotations(self):
        ann = self.dataset.get_annotations(self.current_image_id)
        if ann is None:
            ann_ids = []
        else:
            ann_ids = [x["id"] for x in ann]

        for i in range(len(self.current_annotation_boxes)):
            ann_id = self.current_annotation_ids[i]
            if ann_id in ann_ids:
                # update existing annotation
                category_id = self.dataset.get_category_id_by_name(self.current_annotation_classes[i])
                self.dataset.update_annotation(ann_id, category_id, self.current_annotation_boxes[i])

            else:
                # add new annotation
                bbox = self.current_annotation_boxes[i]
                category_id = self.dataset.get_category_id_by_name(self.current_annotation_classes[i])
                self.dataset.add_annotation(self.current_image_id, category_id, bbox, iscrowd=0)

        # removed annotations
        for i in ann_ids:
            if i not in self.current_annotation_ids:
                self.dataset.remove_annotation(i)

        return

    def load_image(self, idx_offset=1):
        """
        load image

        :param idx_offset - idx shift from current image idx
                            e.g.
                            1 - next image
                            0 - load current image again
                            -1 - prev image
        :return:
        """
        if self.current_image_idx is None:
            self.current_image_idx = 0
        else:
            self.current_image_idx = max(min(self.current_image_idx + idx_offset, len(self.dataset.image_records)-1), 0)

        img_data = self.dataset.get_image(image_index=self.current_image_idx)
        self.current_image_id = img_data['id']

        img_path = img_data['file_name']
        self.current_image = cv2.imread(img_path)
        self.current_annotation_boxes = []
        self.current_annotation_classes = []
        self.current_annotation_ids = []
        annotations = self.dataset.get_annotations(img_data['id'])
        if annotations is not None:
            for ann in annotations:
                # bbox = (ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3])
                self.current_annotation_ids.append(ann['id'])
                self.current_annotation_boxes.append(tuple(ann['bbox']))
                category_name = self.dataset.get_category_name_by_id(ann["category_id"])
                self.current_annotation_classes.append(category_name)

        return


class AnnotatorCoco(Annotator):
    def __init__(self, dataset_root_folder, dataset_file_name, categories=['object'], images_ext='', window_name='Annotator'):
        """
        helper class for annotating images

        :param dataset_root_folder: coco dataset root folder
                    - images should be in dataset_root/images
        :param dataset_file_name: coco dataset json file name
                    - name only - not path
                    - dataset_file assumed to be in dataset_root_folder/annotations
                    - use non-existing file for starting a new coco dataset
        :param list of valid category names to be assigned to annotations
        :param images_ext: specific image extention e.g.  '.png' / '.jpg'
                          This is only used for:
                          1. testing in any images are in the image folder.
                          2. loading images in case of a new dataset
                          if empty '' - all files in the folder will used
        """
        super().__init__(categories, window_name)  # Call base class __init__

        if not Path(dataset_root_folder).exists():
            raise Exception('dataset root folder not found! {}'.format(dataset_root_folder))
        images_folder = Path(dataset_root_folder) / 'images'
        if len(list(images_folder.glob('*'+images_ext))) == 0:
            raise Exception('no images found at {}'.format(images_folder))

        if str(Path(dataset_file_name)) != Path(dataset_file_name).name:
            raise Exception('dataset_file_name must be name only - not a path! got {}'.format(dataset_file_name))
        if Path(dataset_file_name).suffix != '.json':
            raise Exception('dataset_file_name must be a json file! got{}'.format(dataset_file_name))
        dataset_file_path = Path(dataset_root_folder) / 'annotations' / dataset_file_name

        # init dataset
        import coco_dataset_manager as dm
        self.dataset = dm.CocoDatasetManager()
        if dataset_file_path.exists():  # load existing dataset
            self.dataset.load_coco(dataset_file_path)
        else:                           # start new dataset
            self.dataset.set_root(dataset_root_folder)
            self.dataset.set_json_file(dataset_file_name)
            # load all images to the dataset
            image_files = sorted(list(images_folder.glob('*'+images_ext)))
            for imf in image_files:
                self.dataset.add_image(str(imf))

        # add dataset categories to annotator
        dataset_categories = [c['name'] for c in self.dataset.get_categories().values()]
        for c in dataset_categories:
            if c not in self.categories:
                self.categories.append(c)

        self.current_image_id = None

    def save(self):
        self.dataset.save_coco(dataset_root_folder=None, json_file_name=None, copy_images=False, overwrite=True)  # save to current dataset

    def update_current_image_annotations(self):
        """
        save / update current image annotation in dataset

        assume:
        self.current_annotation_boxes: all annotation bboxes
        self.current_annotation_classes: all annotation classes
        self.current_annotation_ids:
                for a new annotation, annotation id is None
                for an existing annotation, annotation id stays the same
        """
        ann = self.dataset.get_image_annotations(self.current_image_id)
        if ann is None:
            dataset_ann_ids = []
        else:
            dataset_ann_ids = [x["id"] for x in ann]

        for i in range(len(self.current_annotation_boxes)):
            ann_id = self.current_annotation_ids[i]
            category_id = self.dataset.get_category_id(self.current_annotation_classes[i])
            if ann_id in dataset_ann_ids:
                # update existing annotation
                bbox_area = self.current_annotation_boxes[i][2] * self.current_annotation_boxes[i][3]
                self.dataset.update_annotation(ann_id, category_id=category_id, bbox=self.current_annotation_boxes[i], area=bbox_area)
            else:
                # add new annotation
                bbox = self.current_annotation_boxes[i]

                dataset_categories = [c['name'] for c in self.dataset.get_categories().values()]
                if self.current_category not in dataset_categories:
                    self.dataset.add_category(self.current_category)
                category_id = self.dataset.get_category_id(self.current_category)

                self.dataset.add_annotation(self.current_image_id, category_id, bbox, iscrowd=0, )

        # removed annotations
        for i in dataset_ann_ids:
            if i not in self.current_annotation_ids:
                self.dataset.remove_annotation(i)

        return

    def load_image(self, idx_offset=1, color_space='RGB'):
        """
        load image

        :param idx_offset - idx shift from current image idx
                            e.g.
                            1 - next image
                            0 - load current image again
                            -1 - prev image
        :return:
        """
        image_ids = sorted(self.dataset.get_image_ids())
        if self.current_image_id is None:
            self.current_image_id = image_ids[0]
        else:
            idx = image_ids.index(self.current_image_id)
            new_idx = min(max(idx + idx_offset, 0), len(image_ids) - 1)
            self.current_image_id = image_ids[new_idx]

        img_data = self.dataset.get_image(self.current_image_id)
        img_path = self.dataset.images_folder / img_data['file_name']
        self.current_image = cv2.imread(img_path)
        if color_space == 'RGB':
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)

        self.current_annotation_boxes = []
        self.current_annotation_classes = []
        self.current_annotation_ids = []
        annotations = self.dataset.get_image_annotations(self.current_image_id)
        if annotations is not None:
            for ann in annotations:
                # bbox = (ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3])
                self.current_annotation_ids.append(ann['id'])
                self.current_annotation_boxes.append(tuple(ann['bbox']))
                category_name = self.dataset.get_category_name(ann['category_id'])
                self.current_annotation_classes.append(category_name)
        return


def automatically_annotate_dataset(coco_dataset_file, model_file, model_image_size=None, confidence_th=0.5):
    """
    helper class for adding automatic annotations for a dataset

    :param coco_dataset_file: images folder
    :param annotations_file: annotations file
    """

    # load coco dataset
    if not os.path.isfile(coco_dataset_file):
        raise Exception('coco dataset file not found: {}'.format(coco_dataset_file))
    import coco_dataset_tmp as cdu
    dataset = cdu.CocoDatasetManager()
    dataset.load_coco(coco_dataset_file)

    # load detector model
    if not os.path.isfile(model_file):
        raise Exception('model file not found: {}'.format(model_file))
    import detector_tracker
    detector_model = detector_tracker.DetectorTracker(model_file, model_image_size,
                                            bbox_roi_intersection_th=0.1, detector_use_cpu=False, verbose=False)
    detector = detector_tracker.SingleFrameDetector(model_file, model_image_size,
                                            bbox_roi_intersection_th=0.1, detector_use_cpu=False, verbose=False)

    # annotate image in dataset
    image_ids = dataset.get_image_ids()
    for img_id in image_ids:
        img_data = dataset.get_image(img_id)
        ann = dataset.get_annotations(img_id)

        img = cv2.imread(img_data['image_file'])

        detector_model.set_detection_roi_polygon(None)
        detector_model.step(img, conf_threshold=confidence_th, nms_iou_threshold=0.5, max_num_detections=10)
        tr = detector_model.get_tracks()

        # see if detections are
    return


# ----------------------------- Main -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="dataset annotator.")
    parser.add_argument("base_folder", type=str, help="dataset base foler. Images must be in base_folder/images !")
    parser.add_argument("annotations_file_name", type=str,
                        help="dataset annotations file name (not path). must be .json!")
    parser.add_argument("--categories", type=str, required=True,
                        help="Comma-separated list of categories (e.g., cat,dog)")
    parser.add_argument("--format", type=str, choices=["folder", "coco"],
                        required=False, default='folder', help="Select the dataset format.")
    args = parser.parse_args()

    base_folder = args.base_folder
    annotations_file_name = args.annotations_file_name
    categories = args.categories.split(',')
    data_format = args.format

    # data_format == 'image_folder'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250421_hadera/20250421_082642'
    # annotations_file_name = "annotations.json"



    # ------------------ 20251214_reshafim ------------------------------ (BGR)
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251214_reshafim'
    # base_folder = os.path.join(exp_folder, '20251214_1322_23/20251214_1327_07/camera_20251214_1427_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '20251214_1232_41/20251214_1233_08/camera_20251214_1333_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '20251214_1232_41/20251214_1235_55/camera_20251214_1336_extracted')  # OK


    # ------------------ 20251208_reshafim ------------------------------ (BGR)
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251208_reshafim'
    # base_folder = os.path.join(exp_folder, '20251208_1235_31/20251208_1239_35/camera_20251208_1339_extracted')  # no target
    # base_folder = os.path.join(exp_folder, '20251208_1246_16/20251208_1247_59/camera_20251208_1348_extracted')  # ends early - OK
    # base_folder = os.path.join(exp_folder, '20251208_1256_17/20251208_1301_26/camera_20251208_1401_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '20251208_1256_17/20251208_1303_25/camera_20251208_1403_extracted')  # ends early - OK
    # base_folder = os.path.join(exp_folder, '20251208_1327_30/20251208_1332_56/camera_20251208_1433_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '20251208_1454_32/20251208_1458_20/camera_20251208_1558_extracted')  # ends early - OK
    # base_folder = os.path.join(exp_folder, '20251208_1454_32/20251208_1500_28/camera_20251208_1600_extracted')  # ends early - OK
    # base_folder = os.path.join(exp_folder, '20251208_1509_37/20251208_1513_17/camera_20251208_1613_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '20251208_1509_37/20251208_1515_50/camera_20251208_1615_extracted')  # passes far - OK
    # base_folder = os.path.join(exp_folder, '20251208_1521_09/20251208_1523_58/camera_20251208_1624_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '20251208_1521_09/20251208_1526_27/camera_20251208_1626_extracted')  # OK


    # ------------------ kfar_galim 04.12.2025 ------------------------------ (BGR)
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251204_kfar_galim'
    # base_folder = os.path.join(exp_folder, '20251204_1453_40/20251204_1454_07/camera_20251204_1554_extracted')  # no target
    # base_folder = os.path.join(exp_folder, '20251204_1511_13/20251204_1511_34/camera_20251204_1611_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '20251204_1511_13/20251204_1515_28/camera_20251204_1615_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '20251204_1526_11/20251204_1527_23/camera_20251204_1627_extracted')  # passes far - OK
    # base_folder = os.path.join(exp_folder, '20251204_1526_11/20251204_1532_30/camera_20251204_1632_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '20251204_1550_34/20251204_1550_54/camera_20251204_1650_extracted')  # passes far - OK
    # base_folder = os.path.join(exp_folder, '20251204_1600_55/20251204_1601_20/camera_20251204_1701_extracted')  # ends early - OK
    # base_folder = os.path.join(exp_folder, '20251204_1600_55/20251204_1603_11/camera_20251204_1703_extracted')  # OK


    # ------------------ kfar_galim 26.11.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251126_kfar_galim'
    # base_folder = os.path.join(exp_folder, '20251126_1507_31/20251126_1510_20/camera_20251126_1610_extracted')  # no target
    # base_folder = os.path.join(exp_folder, '20251126_1507_31/20251126_1511_16/camera_20251126_1611_extracted')  # no target
    # base_folder = os.path.join(exp_folder, '20251126_1507_31/20251126_1513_05/camera_20251126_1613_extracted')  # no target
    # base_folder = os.path.join(exp_folder, '20251126_1523_01/20251126_1526_12/camera_20251126_1626_extracted')  # passes far - OK
    # base_folder = os.path.join(exp_folder, '20251126_1536_04/20251126_1536_50/camera_20251126_1636_extracted')  # ends early - OK
    # base_folder = os.path.join(exp_folder, '20251126_1558_12/20251126_1600_23/camera_20251126_1700_extracted')  # no target
    # base_folder = os.path.join(exp_folder, '20251126_1603_18/20251126_1604_48/camera_20251126_1704_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '20251126_1603_18/20251126_1608_34/camera_20251126_1708_extracted')  # OK


    # ------------------ kfar_galim 27.10.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251027_kfar_galim'
    # base_folder = os.path.join(exp_folder, '20251027_123000/camera_20251027_1230_extracted')  # OK


    # ------------------ lehavim 17.09.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250917_lehavim'
    # base_folder = os.path.join(exp_folder, 'hb003/20250915_1028_24/camera_20250915_1028_extracted')  # no target
    # base_folder = os.path.join(exp_folder, 'hb003/20250917_1747_34/camera_20250917_1747_extracted')  # OK


    # ------------------ lehavim 18.09.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250918_lehavim'
    # base_folder = os.path.join(exp_folder, 'hb002/20250915_1019_36/camera_20250915_1019_extracted')  # no target
    # base_folder = os.path.join(exp_folder, 'hb002/20250918_1040_02/camera_20250918_1040_extracted')  # no target
    # base_folder = os.path.join(exp_folder, 'hb002/20250918_1041_26/camera_20250918_1041_extracted')  # no target
    # base_folder = os.path.join(exp_folder, 'hb002/20250918_1044_49/camera_20250918_1044_extracted')  # OK
    # base_folder = os.path.join(exp_folder, 'hb004/20250918_1451_40/camera_20250918_1451_extracted')  # OK
    # base_folder = os.path.join(exp_folder, 'hb004/20250918_1453_47/camera_20250918_1453_extracted')  # OK
    # base_folder = os.path.join(exp_folder, 'hb004/20250918_1456_00/camera_20250918_1456_extracted')  # OK
    # base_folder = os.path.join(exp_folder, 'pz004/20250918_1154_08/camera_20250918_1154_extracted')  # OK
    # base_folder = os.path.join(exp_folder, 'pz004/20250918_1156_18/camera_20250918_1156_extracted')  # OK
    # base_folder = os.path.join(exp_folder, 'pz004/20250918_1249_56/camera_20250918_1249_extracted')  # OK
    # base_folder = os.path.join(exp_folder, 'pz004/20250918_1252_07/camera_20250918_1252_extracted')  # OK


    # ------------------ kfar_masarik 08.06.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250608_kfar_masarik'
    # base_folder = os.path.join(exp_folder, '2025-06-08_18-04-53/camera_2025_6_8-15_4_56_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '2025-06-08_18-29-51/camera_2025_6_8-15_29_54_extracted')  # passes far - OK
    # base_folder = os.path.join(exp_folder, '2025-06-08_18-51-15/camera_2025_6_8-15_51_18_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '2025-06-08_18-52-56/camera_2025_6_8-15_52_58_extracted')  # passes far - OK
    # base_folder = os.path.join(exp_folder, '2025-06-08_18-53-46/camera_2025_6_8-15_53_49_extracted')  # passes far - OK
    # base_folder = os.path.join(exp_folder, '2025-06-08_18-59-46/camera_2025_6_8-15_59_49_extracted')  # passes far - OK
    # base_folder = os.path.join(exp_folder, '2025-06-08_19-00-25/camera_2025_6_8-16_0_28_extracted')  # passes far - OK
    # base_folder = os.path.join(exp_folder, '2025-06-08_19-08-34/camera_2025_6_8-16_8_48_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '2025-06-08_19-09-33/camera_2025_6_8-16_9_38_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '2025-06-08_19-25-35/camera_2025_6_8-16_25_38_extracted')  # passes far - OK


    # ------------------ hadera 21.04.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250421_hadera'
    # base_folder = os.path.join(exp_folder, '2025-04-21_10-57-38/camera_2025_4_21-7_59_8_extracted')  # no target (almost)
    # base_folder = os.path.join(exp_folder, '2025-04-21_10-59-30/camera_2025_4_21-7_59_41_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '2025-04-21_11-11-25/camera_2025_4_21-8_12_28_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '2025-04-21_11-13-03/camera_2025_4_21-8_13_40_extracted')  # OK
    # base_folder = os.path.join(exp_folder, '2025-04-21_11-24-25/camera_2025_4_21-8_25_32_extracted')  # passes far - OK
    # base_folder = os.path.join(exp_folder, '2025-04-21_11-26-29/camera_2025_4_21-8_26_42_extracted')  # passes far - OK

    # TODO: add  20251005_kfar_galim
    # TODO: add  20250730_kfar_galim


    # data_format = 'coco_dataset'
    # annotations_file_name = "coco_dataset.json"
    # categories = ['rc-plane']
    # color_space = 'BGR'


    print("Controls:")
    print("  D / right arrow: Next image")
    print("  A / left arrow : Prev image")
    print("  C              : Change class name")
    print("  R / del        : Remove box (click)")
    print("  Z              : Clear all boxes")
    print("  S / ESC        : Save and exit")
    print("  mouse MB       : mark / remove bbox")


    if data_format == 'image_folder':
        images_folder = os.path.join(base_folder, "images")  # Folder containing images to annotate
        annotations_file_path = os.path.join(base_folder, annotations_file_name)  # File to save/load annotations
        annotator = AnnotatorStandardRecordFolder(images_folder, annotations_file=annotations_file_path, window_name="Annotator")

    elif data_format == 'coco_dataset':
        annotator = AnnotatorCoco(base_folder, annotations_file_name, categories=categories, window_name="Annotator")

    else:
        raise Exception('invalid dataset format!')

    annotator.load_image(1, color_space=color_space)

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    annotations_output_file = os.path.join(base_folder, "annotations_{}.json".format(datetime_str))  # File to save/load annotations

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", annotator.mouse_callback)

    while True:
        annotator.update_display()
        key = cv2.waitKeyEx(0)
        char = chr(key).lower()

        if key == ord('d') or key == ord('D') or key == 65363:  # d / D / right arrow
            annotator.update_current_image_annotations()
            annotator.save()
            annotator.load_image(1, color_space=color_space)

        elif key == ord('a') or key == ord('A') or key == 65361:  # a / A / left arrow
            annotator.update_current_image_annotations()
            annotator.save()
            annotator.load_image(-1, color_space=color_space)

        elif key == ord('s') or key == ord('S') or key == 27:  # s/ESC - Save and exit
            annotator.update_current_image_annotations()
            annotator.save()
            print("Annotations saved.")
            break

        elif key == ord('c') or key == ord('C'):
            annotator.update_current_category()

        elif key == ord('r') or key == ord('R') or key == 65535:   # r / R / del - delete mode
            print("Click inside a box to remove it...")
            annotator.set_remove_state()

        elif key == ord('z') or key == ord('Z'):
            annotator.clear_current_annotations()
            print("Cleared all annotations.")

        else:
            pass
            # print("invalid key.")

    cv2.destroyAllWindows()
    annotator.update_current_image_annotations()
    annotator.save()
