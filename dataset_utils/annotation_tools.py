import cv2
import os
import json
from datetime import datetime
import numpy as np



class Annotator:
    def __init__(self, window_name='Annotator'):
        """
        helper class for annotating images

        :param window_name: opencv window for draw folder
        """

        # current image variables
        self.current_image_idx = None
        self.current_annotation_boxes = []
        self.current_annotation_classes = []
        self.current_annotation_ids = []
        self.current_class = "object"
        self.current_image = None
        self.current_image_id = None

        # state variables
        self.drawing = False
        self.removing = False
        self.start_point = None

        self.window_name = window_name

        self.model = None

    def load(self):
        raise NotImplementedError("Subclasses must implement load()")

    def save(self, annotations_file):
        self.dataset.save(annotations_file)
        return

    def load_image(self, idx_offset=1):
        raise NotImplementedError("Subclasses must implement load_image()")

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if self.removing:
            cv2.putText(drawn, "REMOVE MODE - Click a box", (30, 30),
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
            for i, (x1, y1, x2, y2) in enumerate(self.current_annotation_boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    del self.current_annotation_boxes[i]
                    del self.current_annotation_classes[i]
                    print("Removed annotation.")
                    break
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
            x2, y2 = x, y
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                self.current_annotation_ids.append(None)
                self.current_annotation_boxes.append((min(x1, x2), min(y1, y2), int(np.round(np.abs(x1 -x2))), int(np.round(np.abs(y1 - y2)))))
                self.current_annotation_classes.append(self.current_class)
            self.update_display()


class AnnotatorStandardRecordFolder(Annotator):
    def __init__(self, record_base_folder, window_name='Annotator'):
        """
        helper class for annotating images

        :param image_folder: images folder
        :param annotations_file: annotations file
        """
        super().__init__(window_name)  # Call base class __init__

        self.base_folder = record_base_folder
        if not os.path.isdir(record_base_folder):
            raise Exception('record base folder {} not found'.format(record_base_folder))

        self.images_folder = os.path.join(self.base_folder, 'images')
        if not os.path.isdir(self.images_folder):
            raise Exception('images folder {} not found'.format(self.images_folder))


        self.dataset_file = os.path.join(self.base_folder, 'coco_annotations.yaml')
        import coco_dataset as cdu
        self.dataset = cdu.CocoDatasetManager()
        if os.path.isfile(self.dataset_file):
            self.dataset.load_coco(coco_dataset_file)

        self.current_image_id = None

    def save_current_image_annotations(self):
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
    def __init__(self, coco_dataset_file, window_name='Annotator'):
        """
        helper class for annotating images

        :param image_folder: images folder
        :param annotations_file: annotations file
        """
        super().__init__(window_name)  # Call base class __init__
        self.dataset_file = coco_dataset_file
        if not os.path.isfile(coco_dataset_file):
            raise Exception('coco dataset file not found: {}'.format(coco_dataset_file))

        import coco_dataset as cdu
        self.dataset = cdu.CocoDatasetManager()
        self.dataset.load_coco(coco_dataset_file)
        self.current_image_id = None

    def save_current_image_annotations(self):
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


def automatically_annotate_dataset(coco_dataset_file, model_file, model_image_size=None, confidence_th=0.5):
    """
    helper class for adding automatic annotations for a dataset

    :param coco_dataset_file: images folder
    :param annotations_file: annotations file
    """

    # load coco dataset
    if not os.path.isfile(coco_dataset_file):
        raise Exception('coco dataset file not found: {}'.format(coco_dataset_file))
    import coco_dataset as cdu
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

    coco_dataset_file =
    model_file =
    automatically_annotate_dataset(coco_dataset_file, model_file, model_image_size=None, confidence_th=0.5)

    # ----------------------------- Configuration -----------------------------
    # choose data mode:
    #   'image_folder' - load images from folder
    #                    load / save annotations using a simple json format
    #                    # useful if you work standalone
    #   'coco_dataset' - load coco dataset
    #                    load / save annotations to coco dataset format
    #                    # useful if you work with the coco dataset format

    data_format = 'coco_dataset'


    if data_format == 'image_folder':
        base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250421_hadera/20250421_082642'
        images_folder = os.path.join(base_folder, "images")  # Folder containing images to annotate
        annotations_file = os.path.join(base_folder, "annotations.json")  # File to save/load annotations
        annotator = AnnotatorStandardRecordFolder(images_folder, annotations_file=annotations_file, window_name="Annotator")


    elif data_format == 'coco_dataset':
        # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250402_kfar_galim/20250402_112645'
        # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250402_kfar_galim/20250402_115652'
        base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250402_kfar_galim/20250402_120223'
        # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250402_kfar_galim/20250402_124124'
        # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250402_kfar_galim/20250402_124905'
        # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250402_kfar_galim/20250402_125011'
        coco_dataset_file = os.path.join(base_folder, 'coco_dataset.json')


        base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/annotations'
        coco_dataset_file = os.path.join(base_folder, 'instances_train.json')

        annotator = AnnotatorCoco(coco_dataset_file, window_name="Annotator")

    else:
        raise Exception('invalid data format!')

    annotator.load_image()

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    annotations_output_file = os.path.join(base_folder, "annotations_{}.json".format(datetime_str))  # File to save/load annotations

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", annotator.mouse_callback)

    print("Controls:")
    print("  D / right arrow: Next image")
    print("  A / left arrow : Prev image")
    print("  C              : Change class name")
    print("  R              : Remove box (click)")
    print("  Z              : Clear all boxes")
    print("  S / ESC        : Save and exit")

    while True:
        annotator.update_display()
        key = cv2.waitKeyEx(0)
        char = chr(key).lower()

        if key == ord('d') or key == 65363:  # d / right arrow
            annotator.save_current_image_annotations()
            annotator.save(annotations_output_file)
            annotator.load_image(1)

        elif key == ord('a') or key == 65361:  # a / left arrow
            annotator.save_current_image_annotations()
            annotator.save(annotations_output_file)
            annotator.load_image(-1)

        elif key == ord('s') or key == 27:  # s/ESC - Save and exit
            annotator.save_current_image_annotations()
            annotator.save(annotations_output_file)
            print("Annotations saved.")
            break

        elif key == ord('c'):
            current_class = input("Enter class name: ")
            annotator.set_current_class(current_class)

        elif key == ord('r'):
            print("Click a box to remove it...")
            annotator.set_remove_state()

        elif key == ord('z'):
            annotator.clear_current_annotations()
            print("Cleared all annotations.")

        else:
            pass
            # print("invalid key.")

    cv2.destroyAllWindows()
    annotator.save_current_image_annotations()
    annotator.save(annotations_output_file)
