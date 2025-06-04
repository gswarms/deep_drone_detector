import cv2
import os
import json
# import coco_dataset_utils as cdu


class AnnotatorSimple:
    def __init__(self, image_folder, window_name='Annotator'):
        """
        helper class for annotating images

        :param image_folder: images folder
        :param annotations_file: annotations file
        """

        self.images_folder = image_folder
        if not os.path.isdir(image_folder):
            raise Exception('images_folder not found: {}'.format(image_folder))
        self.image_files = sorted([f for f in os.listdir(self.images_folder)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png')) ])
        self.annotations = {}

        # current image variables
        self.current_image_idx = None
        self.current_boxes = []
        self.current_classes = []
        self.current_class = "object"
        self.current_image = None

        # state variables
        self.drawing = False
        self.removing = False
        self.start_point = None

        self.window_name = window_name

    def load_annotations(self, annotations_file):
        if os.path.isfile(annotations_file):
            with open(annotations_file, "r") as f:
                self.annotations = json.load(f)
        else:
            raise Exception('annotations_file not found: {}'.format(annotations_file))

    def save_annotations_to_file(self, annotations_file):
        with open(annotations_file, "w") as f:
            json.dump(self.annotations, f, indent=2)

    def save_current_image_annotations(self):
        filename = self.image_files[self.current_image_idx]
        self.annotations[filename] = [{"bbox": list(b), "class": c} for b, c in zip(self.current_boxes, self.current_classes)]

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
            self.current_image_idx = max(min(self.current_image_idx + idx_offset, len(self.image_files)-1), 0)

        filename = self.image_files[self.current_image_idx]
        path = os.path.join(self.images_folder, filename)
        self.current_image = cv2.imread(path)
        self.current_boxes = []
        self.current_classes = []
        if filename in self.annotations:
            for ann in self.annotations[filename]:
                self.current_boxes.append(tuple(ann["bbox"]))
                self.current_classes.append(ann["class"])

    def clear_current_annotations(self):
        self.current_boxes = []
        self.current_classes = []

    def set_current_class(self, class_name):
        self.current_class = class_name

    def set_remove_state(self):
        self.removing = True

    def draw_current_annotations(self):
        annotated = self.current_image.copy()
        for (box, cls) in zip(self.current_boxes, self.current_classes):
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, cls, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated

    @staticmethod
    def draw_annotations(img, boxes, classes):
        for (box, cls) in zip(boxes, classes):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, cls, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img

    def update_display(self, temp_box=None):
        img_copy = self.current_image.copy()
        temp_boxes = self.current_boxes.copy()
        temp_classes = self.current_classes.copy()
        if temp_box:
            temp_boxes.append(temp_box)
            temp_classes.append(self.current_class)

        drawn = self.draw_annotations(img_copy, temp_boxes, temp_classes)

        if self.removing:
            cv2.putText(drawn, "REMOVE MODE - Click a box", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
            for i, (x1, y1, x2, y2) in enumerate(self.current_boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    del self.current_boxes[i]
                    del self.current_classes[i]
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
                self.current_boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                self.current_classes.append(self.current_class)
            self.update_display()



# ----------------------------- Main -----------------------------
if __name__ == "__main__":

    # ----------------------------- Configuration -----------------------------

    # choose data mode:
    #   'image_folder' - load images from folder
    #                    load / save annotations using a simple json format
    #                    # useful if you work standalone
    #   'coco_dataset' - load coco dataset
    #                    load / save annotations to coco dataset format
    #                    # useful if you work with the coco dataset format

    # COCO_DATASET_FILE = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250421_hadera/20250421_082642/coco_dataset.json'

    base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250421_hadera/20250421_082642'
    images_folder = os.path.join(base_folder, "images")  # Folder containing images to annotate
    annotations_file = os.path.join(base_folder, "annotations.json")  # File to save/load annotations


    annotator = AnnotatorSimple(images_folder, window_name="Annotator")
    if annotations_file is not None and os.path.isfile(annotations_file):
        annotator.load_annotations(annotations_file)
    annotator.load_image()

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", annotator.mouse_callback)

    print("Controls:")
    print("  A/D           : Prev/Next image")
    print("  Shift+A/D     : Jump -10/+10 images")
    print("  C             : Change class")
    print("  R             : Remove box (click)")
    print("  Z             : Clear all boxes")
    print("  S             : Save and exit")
    print("  ESC           : Exit without saving")
    print("  +/- or Scroll : Zoom in/out (centered at mouse)")
    print("  Arrows        : Pan")

    while True:
        annotator.update_display()
        key = cv2.waitKeyEx(0)
        char = chr(key).lower()

        if key == ord('d') or key == 65363:  # d / right arrow
            annotator.save_current_image_annotations()
            annotator.save_annotations_to_file(annotations_file)
            annotator.load_image(1)

        elif key == ord('a') or key == 65361:  # a / left arrow
            annotator.save_current_image_annotations()
            annotator.save_annotations_to_file(annotations_file)
            annotator.load_image(-1)

        elif key == ord('s') or key == 27:  # s/ESC - Save and exit
            annotator.save_current_image_annotations()
            annotator.save_annotations_to_file(annotations_file)
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
    annotator.save_annotations_to_file(annotations_file)
