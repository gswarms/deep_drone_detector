import cv2
import os
import json

# ----------------------------- Configuration -----------------------------
base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw/20250421_hadera/20250421_082642'
IMAGE_FOLDER = os.path.join(base_folder ,"images")  # Folder containing images to annotate
ANNOTATIONS_FILE = os.path.join(base_folder, "coco_data.json")   # File to save/load annotations

# ----------------------------- Globals -----------------------------
annotations = {}
image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

current_image_idx = 0
current_boxes = []
current_classes = []
current_class = "object"
current_image = None
drawing = False
removing = False
start_point = None

zoom = 1.0
pan_offset = [0.0, 0.0]
PAN_STEP = 50
ZOOM_STEP = 0.1
MAX_ZOOM = 5.0
MIN_ZOOM = 0.2

# ----------------------------- Annotation Helpers -----------------------------
def load_annotations():
    global annotations
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, "r") as f:
            annotations = json.load(f)

def save_annotations():
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations, f, indent=2)

def save_current_image_annotations():
    filename = image_files[current_image_idx]
    annotations[filename] = [
        {"bbox": list(b), "class": c}
        for b, c in zip(current_boxes, current_classes)
    ]

def load_current_image():
    global current_image, current_boxes, current_classes, pan_offset, zoom
    filename = image_files[current_image_idx]
    path = os.path.join(IMAGE_FOLDER, filename)
    current_image = cv2.imread(path)
    current_boxes = []
    current_classes = []
    zoom = 1.0
    pan_offset = [0.0, 0.0]
    if filename in annotations:
        for ann in annotations[filename]:
            current_boxes.append(tuple(ann["bbox"]))
            current_classes.append(ann["class"])

# ----------------------------- Drawing -----------------------------
def draw_annotations(img, boxes, classes):
    annotated = img.copy()
    for (box, cls) in zip(boxes, classes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, cls, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated

def update_display(temp_box=None):
    img_copy = current_image.copy()
    temp_boxes = current_boxes.copy()
    temp_classes = current_classes.copy()

    if temp_box:
        temp_boxes.append(temp_box)
        temp_classes.append(current_class)

    drawn = draw_annotations(img_copy, temp_boxes, temp_classes)

    if removing:
        cv2.putText(drawn, "REMOVE MODE - Click a box", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Annotator", drawn)

# ----------------------------- Mouse Callback -----------------------------
def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, removing, zoom, pan_offset

    if removing and event == cv2.EVENT_MBUTTONDOWN:
        for i, (x1, y1, x2, y2) in enumerate(current_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                del current_boxes[i]
                del current_classes[i]
                print("Removed annotation.")
                break
        removing = False
        update_display()
        return

    if event == cv2.EVENT_MBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        update_display(temp_box=(start_point[0], start_point[1], x, y))

    elif event == cv2.EVENT_MBUTTONUP and drawing:
        drawing = False
        x1, y1 = start_point
        x2, y2 = x, y
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            current_boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            current_classes.append(current_class)
        update_display()

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    if not image_files:
        print("No images found in", IMAGE_FOLDER)
        exit(1)

    load_annotations()
    load_current_image()

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", mouse_callback)

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
        update_display()
        key = cv2.waitKeyEx(0)
        char = chr(key & 0xFF).lower()

        if char == 'd':
            save_current_image_annotations()
            save_annotations()
            if key & 0x10000:  # Shift+D
                current_image_idx = min(len(image_files) - 1, current_image_idx + 10)
            else:
                current_image_idx = min(len(image_files) - 1, current_image_idx + 1)
            load_current_image()

        elif char == 'a':
            save_current_image_annotations()
            save_annotations()
            if key & 0x10000:  # Shift+A
                current_image_idx = max(0, current_image_idx - 10)
            else:
                current_image_idx = max(0, current_image_idx - 1)
            load_current_image()

        elif char == 's':  # Save and exit
            save_current_image_annotations()
            save_annotations()
            print("Annotations saved.")
            break

        elif char == 'c':
            current_class = input("Enter class name: ")

        elif char == 'r':
            print("Click a box to remove it...")
            removing = True

        elif char == 'z':
            current_boxes.clear()
            current_classes.clear()
            print("Cleared all annotations.")

        elif key == 27:  # ESC
            print("Exited without saving.")
            break

    cv2.destroyAllWindows()
    save_annotations()
