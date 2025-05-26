import os
import shutil
import random
from collections import defaultdict


def get_label_path(img_path):
    name = os.path.splitext(os.path.basename(img_path))[0]
    scenario = os.path.basename(os.path.dirname(img_path))
    return os.path.join(base_lbl_dir, scenario, name + '.txt')


if __name__ == '__main__':

    dataset_baseline_dir = '/home/roee/Downloads/Drone Dataset.v2i.yolov8/dataset_baseline'
    dataset_res_dir = '/home/roee/Downloads/Drone Dataset.v2i.yolov8/dataset_20250522_coco'


    # CONFIG
    base_img_dir = 'images'
    base_lbl_dir = 'labels'
    output_dir = 'dataset'
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    seed = 42
    random.seed(seed)


    scenario_dirs = [d for d in os.listdir(base_img_dir) if os.path.isdir(os.path.join(base_img_dir, d))]
    splits = {'train': [], 'val': [], 'test': []}

    # Collect images per scenario
    for scenario in scenario_dirs:
        img_dir = os.path.join(base_img_dir, scenario)
        imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        random.shuffle(imgs)

        total = len(imgs)
        n_train = int(total * split_ratios['train'])
        n_val = int(total * split_ratios['val'])

        splits['train'].extend(imgs[:n_train])
        splits['val'].extend(imgs[n_train:n_train+n_val])
        splits['test'].extend(imgs[n_train+n_val:])

    # Copy files
    for split, file_list in splits.items():
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, subdir, split), exist_ok=True)
        for img_path in file_list:
            filename = os.path.basename(img_path)
            scenario = os.path.basename(os.path.dirname(img_path))
            lbl_path = get_label_path(img_path)

            dst_img_path = os.path.join(output_dir, 'images', split, filename)
            dst_lbl_path = os.path.join(output_dir, 'labels', split, os.path.splitext(filename)[0] + '.txt')

            shutil.copy(img_path, dst_img_path)
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, dst_lbl_path)
            else:
                open(dst_lbl_path, 'w').close()
