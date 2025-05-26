import os
import json
import cv2
import glob

def convert_yolo_to_coco(yolo_img_dir, yolo_label_dir, class_list, output_json_path):
    categories = [{"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(class_list)]

    images = []
    annotations = []
    ann_id = 1
    img_id = 0

    for filename in os.listdir(yolo_label_dir):
        if not filename.endswith('.txt'):
            continue

        image_name = filename.replace('.txt', '.jpg')  # or .png
        image_path = os.path.join(yolo_img_dir, image_name)
        label_path = os.path.join(yolo_label_dir, filename)

        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        images.append({
            "file_name": image_name,
            "height": h,
            "width": w,
            "id": img_id
        })

        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls_id, x, y, bw, bh = map(float, line.strip().split())
                x_min = int((x - bw / 2) * w)
                y_min = int((y - bh / 2) * h)
                width = int(bw * w)
                height = int(bh * h)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_id),
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                ann_id += 1

        img_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, 'w') as out_file:
        json.dump(coco_dict, out_file, indent=4)



# def split_yolo_dataset():
#     import glob
#     import re
#     import shutil
#
#     base_yolo_dir = '/home/roee/Downloads/Drone Dataset.v2i.yolov8'
#
#     yolo_img_dirs = [os.path.join(base_yolo_dir, 'test', 'images'),
#                      os.path.join(base_yolo_dir, 'train', 'images'),
#                      os.path.join(base_yolo_dir, 'valid', 'images')]
#
#     yolo_label_dirs = [os.path.join(base_yolo_dir, 'test', 'labels'),
#                      os.path.join(base_yolo_dir, 'train', 'labels'),
#                      os.path.join(base_yolo_dir, 'valid', 'labels')]
#
#     for d in yolo_img_dirs:
#         img_files = glob.glob(os.path.join(d,'*.jpg'))
#         for img_file in img_files:
#
#             sp = re.split(r'[_,-]+', os.path.basename(img_file))
#
#             scen_name = '{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(int(sp[1]), int(sp[2]), int(sp[3]), int(sp[4]), int(sp[5]), int(sp[6]))
#             res_img_name = scen_name + '_' + sp[7] + '.jpg'
#
#             res_folder = os.path.join(base_yolo_dir, scen_name, 'images')
#             if not os.path.isdir(res_folder):
#                 os.makedirs(res_folder)
#             shutil.copy(os.path.join(d,img_file), os.path.join(res_folder,res_img_name))
#
#
#     for d in yolo_label_dirs:
#         lbl_files = glob.glob(os.path.join(d,'*.txt'))
#         for lbl_file in lbl_files:
#             sp = re.split(r'[_,-]+', os.path.basename(lbl_file))
#
#             scen_name = '{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(int(sp[1]), int(sp[2]), int(sp[3]), int(sp[4]), int(sp[5]), int(sp[6]))
#             res_name = scen_name + '_' + sp[7] + '.txt'
#
#             res_folder = os.path.join(base_yolo_dir, scen_name, 'labels')
#             if not os.path.isdir(res_folder):
#                 os.makedirs(res_folder)
#             shutil.copy(os.path.join(d,lbl_file), os.path.join(res_folder,res_name))


if __name__ == '__main__':

    # split_yolo_dataset()

    # Example usage:
    yolo_base_dir = '/home/roee/Downloads/Drone Dataset.v2i.yolov8/dataset_baseline'
    class_list = ['fixed_wing']  # or load from .names file

    scenario_dirs = glob.glob(os.path.join(yolo_base_dir,'*','*'))

    for scen_dir in scenario_dirs:
        yolo_img_dir = os.path.join(scen_dir, 'images')
        yolo_label_dir = os.path.join(scen_dir, 'labels')
        if not os.path.isdir(yolo_img_dir) or not os.path.isdir(yolo_label_dir):
            raise Exception('folder not found!')
        output_json_path = os.path.join(scen_dir, 'data.json')
        convert_yolo_to_coco(yolo_img_dir, yolo_label_dir, class_list, output_json_path)