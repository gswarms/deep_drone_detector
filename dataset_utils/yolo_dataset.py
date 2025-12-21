import os
import yaml
import cv2
from pathlib import Path
import shutil
import coco_dataset_manager


def load_yolo_dataset(yolo_dir: str, verify_images=True):
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

    dataset = common_dataset.DatasetManager()

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
        category_id = dataset.add_category(cn, 'none')
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
        image_id = dataset.add_image(str(image_file.resolve()), width, height)

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
                    coco_bbox = _yolo_to_coco_bbox(x, y, w, h, width, height)
                    dataset.add_annotation(image_id, remapped_category_id,
                                        coco_bbox, iscrowd=0)

    # sort images by file name
    dataset._sort_images()

    # verify images exist
    if verify_images:
        missing_images = dataset._verify_images_exist()
        if len(missing_images) > 0:
            for mi in missing_images:
                print('image file missing: {}'.format(mi))
            raise Exception('missing image files')

    return dataset


def save_yolo_dataset(dataset:common_dataset.DatasetManager, yolo_dataset_dir: str, verbose=True):

    """
    save yolo dataset

    :param dataset: common_dataset.DatastManager
    :param yolo_dir:
    :return:
    """

    yolo_dataset_dir = os.path.abspath(yolo_dataset_dir)
    os.makedirs(yolo_dataset_dir, exist_ok=True)

    yolo_images_dir = os.path.join(yolo_dataset_dir, 'images')
    os.makedirs(yolo_images_dir, exist_ok=True)

    yolo_labels_dir = os.path.join(yolo_dataset_dir, 'labels')
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # copy images to images folder
    for img in dataset.image_records:
        img_file_name = os.path.basename(img['file_name'])
        shutil.copyfile(img['file_name'], os.path.join(yolo_images_dir, img_file_name))

        # write annotation to annotation files
        labels_file_name = os.path.join(yolo_labels_dir, img_file_name[:-4] + '.txt')
        annotations = dataset.get_annotations(img['id'])
        if annotations is not None:
            with f as open(annotations,'w'):

                for ann in annotations:
                    bbox = ann['bbox']
                    yolo_bbox = _coco_to_yolo_bbox(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

    if verbose:
        print(f"YOLO dataset exported to: {yolo_dataset_dir}")

def _yolo_to_coco_bbox(x, y, w, h, img_w, img_h):
    """
    convert from yolo bbox format to coco bbox format

    yolo bbox (x, y, w, h) format:
    x,y - bbox center
    w,h - bbox width and height
    normalize coordinates where x in [0,1] and y in [0,1]

    coco bbox (x, y, w, h) format:
    x,y - top left
    w,h - bbox width and height
    pixel coordinates where x in [0,image width] and y in [0,image height]

    in both cases, x-right and y-down, and image origin is at the image top left pixel
    """
    xtl = x - w / 2
    ytl = y - h / 2
    return [xtl * img_w, ytl * img_h, w * img_w, h * img_h]


def _coco_to_yolo_bbox(x, y, w, h, img_w, img_h):
    """
    convert from coco bbox format to yolo bbox format

    yolo bbox (x, y, w, h) format:
    x,y - bbox center
    w,h - bbox width and height
    normalize coordinates where x in [0,1] and y in [0,1]

    coco bbox (x, y, w, h) format:
    x,y - top left
    w,h - bbox width and height
    pixel coordinates where x in [0,image width] and y in [0,image height]

    in both cases, x-right and y-down, and image origin is at the image top left pixel
    """
    xc = x + w / 2
    yc = x + h / 2
    return [xc / img_w, yc / img_h, w / img_w, h / img_h]


if __name__ == '__main__':

    import glob

    # convert each yolo scenario to a coco scenario
    dataset_base_dir = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw'
    dataset_scearios = glob.glob(os.path.join(dataset_base_dir, '*', '*'))
    for scen_path in dataset_scearios:
        manager = CocoDatasetManager()
        manager.load_yolo(scen_path)
        coco_dataset_file = os.path.join(scen_path, 'coco_dataset.json')
        manager.save(coco_dataset_file, verbose=True)

    # join coco datasets to one dataset
    manager = CocoDatasetManager()
    dataset_base_dir = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_rcplane_raw'
    dataset_scearios = glob.glob(os.path.join(dataset_base_dir, '*', '*'))
    for scen_path in dataset_scearios:
        coco_dataset_file = os.path.join(scen_path, 'coco_dataset.json')
        manager.load_coco(coco_dataset_file)
    manager.save(os.path.join(dataset_base_dir, 'coco_dataset.json'), verbose=True)