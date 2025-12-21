import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional, Dict, Any
import hashlib
import shutil
from datetime import datetime
import time
import numpy as np


class CocoDatasetManager:
    """
    This object is meant for managing coco dataset.
    - load / save dataset
    - merge datasets
    - add / remove images
    - add / remove / update annotation
    - add / remove / remap categories
    - add / remove / update image metadata
    - Export / import from YOLO format

    We follow the coco format as described in the Readme.md
    """
    def __init__(self):

        self.root_folder = None
        self.images_folder = None
        self.json_path = None

        # Core COCO tables as pandas DataFrames
        self.df_images = pd.DataFrame(columns=[
            "id", "file_name", "width", "height", "metadata"
        ]).astype({
            "id": "int64",
            "file_name": "str",
            "width": "int64",
            "height": "int64",
            "metadata": "object"
        })

        self.df_annotations = pd.DataFrame(columns=[
            "id", "image_id", "category_id", "bbox", "segmentation", "area", "iscrowd", "metadata"
        ]).astype({
            "id": "int64",
            "image_id": "int64",
            "category_id": "int64",
            "bbox": "object",
            "segmentation": "object",
            "area": "float64",
            "iscrowd": "int64",
            "metadata": "object"
        })

        self.df_categories = pd.DataFrame(columns=["id", "name", "supercategory"]).astype({
            "id": "int64",
            "name": "str",
            "supercategory": "str"
        })

        # For tracking next IDs
        self._next_image_id = 1
        self._next_annotation_id = 1
        self._next_category_id = 1

        self._image_get_id = 0

    # --------------------
    # manual set
    # --------------------
    def set_root(self, root_folder, images_folder=None):
        """
        set root folder
        :return:
        """
        self.root_folder = Path(root_folder)
        if images_folder is None:
            self.images_folder = self.root_folder / "images"
        else:
            self.images_folder = Path(images_folder)

    def set_json_file(self, json_file_name):
        """
        set json file name
        this is use full for:
            1. starting a new dataset
            2. loading one json file, and saving to another
        :return:
        """
        self.json_path = self.root_folder / 'annotations' / json_file_name

    # --------------------
    # Load / Save
    # --------------------
    def load_coco(self, json_path: str, verify_image_files=False):
        """
        load coco dataset

        Assumptions:
        1. dataset folder structure:
            dataset_root_folder/
            ├── annotations/
            │   └── some_name.json
            ├── images/            # Tracking algorithms (e.g., OpenCV, DeepSORT)
            │   ├── img1.png
            │   ├── ...
            │   └── img1000.png
            or json file is right under dataset_root_folder without 'annotations' subfolder
            subfolders under images are supported!

        2. image file path in the json file are relative to the /images subfolder

        :param json_path:
        :param verify_image_files: make sure all image files exist
        :return:
        """
        if not Path(json_path).exists():
            raise Exception('dataset json path {} not found!'.format(json_path))

        with open(json_path, "r") as f:
            data = json.load(f)

        self.json_path = Path(json_path)
        if self.json_path.parent.name == 'annotations':
            self.root_folder = self.json_path.parent.parent
        else:
            raise Exception('dataset json file must be in dataset_root/annotation folder!')

        # Load images
        self.images_folder = self.root_folder / 'images'

        data_images = data.get("images", [])
        if len(data_images) > 0:
            self.df_images = pd.DataFrame(data_images)

        if not self.df_images.empty and "metadata" not in self.df_images.columns:
            self.df_images["metadata"] = [{} for _ in range(len(self.df_images))]

        if verify_image_files:
            for i, row in self.df_images.iterrows():
                relative_img_file_path = Path(row["file_name"])
                img_file_path = self.images_folder / relative_img_file_path
                if not os.path.isfile(img_file_path):
                    raise Exception('image file: {} not found!'.format(img_file_path))

        self._next_image_id = self.df_images["id"].max() + 1 if not self.df_images.empty else 1

        # Load annotations
        data_annotations = data.get("annotations", [])
        if len(data_annotations) > 0:
            self.df_annotations = pd.DataFrame(data_annotations)

        if not self.df_annotations.empty and "metadata" not in self.df_annotations.columns:
            self.df_annotations["metadata"] = [{} for _ in range(len(self.df_annotations))]
        self._next_annotation_id = self.df_annotations["id"].max() + 1 if not self.df_annotations.empty else 1

        # Load categories
        data_categories = data.get("categories", [])
        if len(data_categories) > 0:
            self.df_categories = pd.DataFrame(data_categories)
        self._next_category_id = self.df_categories["id"].max() + 1 if not self.df_categories.empty else 1


    def save_coco(self, dataset_root_folder=None, json_file_name=None, copy_images: bool = True, overwrite=False):
        """
        Save the COCO dataset to a JSON file.

        :param dataset_root_folder: new dataset root folder
                                    None = save to the current root folder
                                   * it's Ok to choose a new root dir without a new json file name (the name will stay the same)
        :param json_file_name: json file name.
                               None = save to the current json file
                               * it's Ok to choose a new json file name with the same dataset root
                                 images will not be copied since they already exist
        :param copy_images: True  - copy all images to the new dataset folder.
                                    All images will be copied relative to the new images folder.
                            False - keep all images in place
                                    Image paths in the new json will be their current location, but relative to the new images folder.
        :param overwrite: True - overwrite existing json file
                          False - make a backup copy of the json file is it already exists
        """

        if dataset_root_folder is not None:
            dataset_root_folder = Path(dataset_root_folder)
            images_folder = dataset_root_folder / "images"
            if copy_images:
                images_folder.mkdir(parents=True, exist_ok=True)
        else:
            dataset_root_folder = self.root_folder
            images_folder = self.images_folder

        annotation_folder = dataset_root_folder / "annotations"
        annotation_folder.mkdir(parents=True, exist_ok=True)

        if json_file_name is not None:
            json_file_name = Path(json_file_name)
            if json_file_name.suffix != '.json':
                raise Exception('json file name must have a .json suffix! got {}'.format(json_file_name))
        else:
            json_file_name = self.json_path.name
        json_file_path = annotation_folder / json_file_name

        # backup json file if we want to rewrite over it
        if json_file_path == self.json_path and self.json_path.exists() and overwrite==False:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_json = self.json_path.with_name(self.json_path.stem + "_backup_" + date_str + self.json_path.suffix)
            shutil.copy2(self.json_path, backup_json)

        coco_dict = {"images": [], "annotations": [], "categories": []}

        # Images
        for _, row in self.df_images.iterrows():
            orig_path = self.images_folder / Path(row["file_name"])

            if copy_images:
                new_path = images_folder / Path(row["file_name"])
                if not new_path.exists():
                    new_path.parent.mkdir(parents=True, exist_ok=True)  # make new subfolders if needed
                    shutil.copy2(orig_path, new_path)
            else:
                new_path = orig_path.relative_to(images_folder)

            tmp_dict = {
                "id": row["id"],
                "file_name": str(new_path),
                "width": row["width"],
                "height": row["height"]
            }
            if len(row["metadata"]) > 0:
                tmp_dict["metadata"] = row["metadata"]
            coco_dict["images"].append(tmp_dict)

        # Annotations
        coco_dict["annotations"] = self.df_annotations.to_dict(orient="records")
        for ann in coco_dict["annotations"]:
            if not ann.get("metadata"):  # empty dict, None, or missing
                ann.pop("metadata", None)

        # Categories
        coco_dict["categories"] = self.df_categories.to_dict(orient="records")

        # Save JSON
        with open(json_file_path, "w") as f:
            json.dump(coco_dict, f, indent=4)

    # --------------------
    # Merge another dataset
    # --------------------
    def merge_dataset(self, other: "CocoDatasetManager", merge_hash_duplicates=False,
                      merge_overlapping_bbox_annotations=True, verify_other_images=True, bbox_overlap_iou_th = 0.2,
                      verbose=True):
        """
        Merge another CocoDatasetManager into self.

        :param other - other CocoDatasetManager to merge
        :param merge_hash_duplicates - bool. Merge duplicate images by hash
                    True - use image hash to find and merge duplicate images
                           may be time-consuming!
                    False - don't merge duplicate images

        :param merge_overlapping_bbox_annotations - bool. Merge annotations from different datasets
                                                         on the same image and with overlapping bboxs
        :param verify_other_images - bool. except if other image files do not exist
        :param bbox_overlap_iou_th -[0,1] threshold for merging overlapping annotations on trhe same image.
        """

        if verbose:
            print('merging datasets:')
        # ---------------------------
        # Step 1: Merge categories with remapping
        # ---------------------------
        if verbose:
            print('   category mapping: {other cat id: mapped cat id in self}')
        cat_map = {}  # key = other cat id: value = corresponding self cat id
        for _, row in other.df_categories.iterrows():
            existing = self.df_categories[self.df_categories["name"] == row["name"]]
            if not existing.empty:
                cat_map[row["id"]] = existing.iloc[0]["id"]
            else:
                new_id = self._next_category_id
                self._next_category_id += 1
                self.df_categories.loc[len(self.df_categories)] = [new_id, row["name"], row["supercategory"]]
                cat_map[row["id"]] = new_id
        if verbose:
            print('   {}'.format(cat_map))
            print('merging images')

        # ---------------------------
        # Step 2: Compute hashes + paths for existing images
        # ---------------------------
        if verbose and merge_hash_duplicates:
                print('   calculating self image hashes')

        existing_hashes = {}
        existing_paths = {}
        n_images = self.df_images.shape[0]
        t0 = time.monotonic()
        for i, row in self.df_images.iterrows():
            img_path = (self.images_folder / Path(row["file_name"])).resolve()
            existing_paths[img_path] = row["id"]
            if merge_hash_duplicates:
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        h = hashlib.md5(f.read()).hexdigest()
                    existing_hashes[h] = row["id"]
                print(f"   progress: {i}/{n_images} ({time.monotonic()-t0}sec)", end='\r')

        # ---------------------------
        # Step 3: Merge images
        # ---------------------------
        image_map = {}  # key = other img id: value = corresponding self img id
        duplicate_image_ids = []
        n_images = other.df_images.shape[0]
        t0 = time.monotonic()
        for i, row in other.df_images.iterrows():
            other_relative_img_path = other.images_folder / Path(row["file_name"])
            other_abs_img_path = other_relative_img_path.resolve()

            if verify_other_images and not other_abs_img_path.exists():
                raise Exception('other dataset image {} not found!'.format(other_abs_img_path))

            # detect path duplicates
            img_match_found = False
            if other_abs_img_path in existing_paths:
                image_map[row["id"]] = existing_paths[other_abs_img_path]
                img_match_found = True
                duplicate_image_ids.append(row["id"])

            # detect hash duplicates
            if merge_hash_duplicates and not img_match_found:
                if other_abs_img_path.exists():
                    with open(other_abs_img_path, "rb") as f:
                        h = hashlib.md5(f.read()).hexdigest()
                    if h in existing_hashes:
                        # Duplicate found, reuse existing image id
                        image_map[row["id"]] = existing_hashes[h]
                        img_match_found = True
                        duplicate_image_ids.append(row["id"])

            # Otherwise, add new image
            if not img_match_found:
                new_id = self._next_image_id
                self._next_image_id += 1
                image_map[row["id"]] = new_id

                # Transform path relative to self
                new_path_abs = (self.images_folder / other_abs_img_path.name).resolve()

                if new_path_abs in self.df_images['file_name'].values:
                    new_name = other_abs_img_path.name.stem + '_' + other_abs_img_path.name.suffix
                    new_path_abs = (self.images_folder / new_name).resolve()

                # copy image
                if not self.images_folder.exists():
                    os.makedirs(self.images_folder)
                shutil.copy2(other_abs_img_path, new_path_abs)

                # add new image to self
                new_path_relative = new_path_abs.relative_to(self.images_folder)
                self.df_images.loc[len(self.df_images)] = [
                    new_id, str(new_path_relative), row["width"], row["height"], row.get("metadata", {})
                ]

            if verbose:
                print(f"   progress: {i}/{n_images} ({time.monotonic() - t0}sec)", end='\r')

        # ---------------------------
        # Step 4: Merge annotations with remapped IDs
        # ---------------------------
        for _, row in other.df_annotations.iterrows():
            other_img_id = row["image_id"]

            # resolve duplicate image annotations
            duplicate_annotation = False
            if other_img_id in duplicate_image_ids:
                self_img_id = image_map[row["image_id"]]
                # get all annotations from self
                self_anns = self.get_image_annotations(self_img_id)
                # test bbox overlap
                max_iou_score = 0
                for ann in self_anns:
                    iou_score = _bbox_iou(row["bbox"], ann['bbox'])
                    max_iou_score = max(max_iou_score,iou_score)

                if max_iou_score >= bbox_overlap_iou_th:
                    duplicate_annotation = True

            if not duplicate_annotation:
                new_id = self._next_annotation_id
                self._next_annotation_id += 1
                self.df_annotations.loc[len(self.df_annotations)] = [
                    new_id,
                    image_map[row["image_id"]],
                    cat_map[row["category_id"]],
                    row.get("bbox", []),
                    row.get("segmentation", []),
                    row.get("area", 0.0),
                    row.get("iscrowd", 0),
                    row.get("metadata", {})
                ]

    # --------------------
    # Images
    # --------------------
    def add_image(self, file_path: str, image_size: tuple[int, int] | list[int] = None, metadata: Optional[dict] = None) -> int:
        """
        add image

        :param file_path:  common practice is path relative to the images folder, but this is not enforced!
        :param image_size: (width, height)
        :param metadata: dict
        :return:
        """

        if self.root_folder is None or self.images_folder is None:
            raise Exception('root folder must be set before adding images')

        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Image {str(file_path)} does not exist.")

        # test if image already in dataset
        file_path_relative = os.path.relpath(file_path, self.images_folder)
        if file_path_relative in self.df_images['file_name'].values:
            raise Exception('duplicate image added: {}'.format(file_path))

        with Image.open(file_path) as im:
            width_img, height_img = im.size
        if image_size is None:
            width = width_img
            height = height_img
        else:
            width, height = image_size
        if width != width_img or height != height_img:
            raise Exception('image size ({}x{}) does not correspond to actual image size ({}x{})'.format(width, height, width_img, height_img))

        img_id = self._next_image_id
        self._next_image_id += 1
        self.df_images.loc[len(self.df_images)] = [img_id, str(file_path_relative), width, height, metadata or {}]
        return img_id

    def remove_image(self, image_id: int):
        img_found = image_id in self.df_images["id"].values
        self.df_images = self.df_images[self.df_images["id"] != image_id]
        self.df_annotations = self.df_annotations[self.df_annotations["image_id"] != image_id]
        return img_found

    def update_image_metadata(self, image_id: int, metadata: Optional[dict] = None) -> int:
        if image_id not in self.df_images["id"].values:
            raise ValueError(f"image id {image_id} not found!")
        else:
            self.df_images.loc[self.df_images["id"] == image_id, "metadata"] = [metadata]

    def get_image(self, image_id: int) -> dict:
        row = self.df_images[self.df_images["id"] == image_id]
        if row.empty:
            return None
        self._image_get_id = row.iloc[0]["id"]
        return row.iloc[0].to_dict()

    def get_image_ids(self) -> List[int]:
        return self.df_images["id"].tolist()

    def get_next_image_id(self, current_image_id=None) -> Optional[int]:
        """
        get the next image id

        Behavior:
        1. If current_image_id is not None - get the next image id after current_image_id
        2. If current_image_id is None     - get the next image id after the last image we got with get_image

        * if no images, return None
        * If we still didn't get any image, next image id will be the first image
        * return None after last image
        * if current_image_id doesn't exist we still find the next image id!

        :param current_image_id: Get the image id that comes after this id
                                 if None - current id will be the last image id that we got.
        :return:
        """
        ids = sorted(self.get_image_ids())

        if current_image_id is None:
            current_image_id = self._image_get_id

        for i in ids:
            if i > current_image_id:
                return i
        return None

    # --------------------
    # Annotations
    # --------------------
    def add_annotation(self, image_id: int, category_id: int, bbox: list, segmentation: Optional[list] = None,
                       area: Optional[float] = None, iscrowd: int = 0, metadata: Optional[dict] = None) -> int:
        if image_id not in self.df_images["id"].values:
            raise ValueError(f"Image ID {image_id} does not exist.")
        if category_id not in self.df_categories["id"].values:
            raise ValueError(f"Category ID {category_id} does not exist.")
        ann_id = self._next_annotation_id
        self._next_annotation_id += 1
        area = area if area is not None else bbox[2] * bbox[3]
        self.df_annotations.loc[len(self.df_annotations)] = [
            ann_id, image_id, category_id, bbox, segmentation or [], area, iscrowd, metadata or {}
        ]
        return ann_id

    def remove_annotation(self, annotation_id: int):
        self.df_annotations = self.df_annotations[self.df_annotations["id"] != annotation_id]

    def update_annotation(self, annotation_id: int, **kwargs):
        if annotation_id in self.df_annotations["id"].values:
            idx = self.df_annotations.index[self.df_annotations["id"] == annotation_id][0]
            for key, value in kwargs.items():
                    if key in self.df_annotations.columns:
                        self.df_annotations.at[idx, key] = value
                    else:
                        raise Exception('key {} is not found in annotations'.format(key))
        else:
            raise Exception('annotation id {} does not exist!'.format(annotation_id))

    def get_annotation(self, annotation_id: int) -> dict:
        row = self.df_annotations[self.df_annotations["id"] == annotation_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_image_annotations(self, image_id: int) -> List[dict]:
        rows = self.df_annotations[self.df_annotations["image_id"] == image_id]
        return rows.to_dict(orient="records")

    # --------------------
    # Categories
    # --------------------
    def add_category(self, name: str, supercategory: str = "") -> int:
        if name in self.df_categories["name"].values:
            raise ValueError(f"Category {name} already exists.")
        cat_id = self._next_category_id
        self._next_category_id += 1
        self.df_categories.loc[len(self.df_categories)] = [cat_id, name, supercategory]
        return cat_id

    def update_category(self, category_id: int, name: str, supercategory: str = ""):
        if category_id not in self.df_categories["id"].values:
            raise ValueError(f"Category {name} not found!")
        elif (self.df_categories["name"] == name).any():
            raise Exception(f"Category {name} with superclass {supercategory} already exists!")
        else:
            self.df_categories.loc[self.df_categories["id"] == category_id, "supercategory"] = supercategory
            self.df_categories.loc[self.df_categories["id"] == category_id, "name"] = name

    def remove_category(self, category_id: int) -> bool:
        if category_id in self.df_categories["id"].values:
            self.df_categories = self.df_categories[self.df_categories["id"] != category_id]
            self.df_annotations = self.df_annotations[self.df_annotations["category_id"] != category_id]
            ret = True
        else:
            ret = False
        return ret

    def remap_category_name(self, old_name: str, new_name: str):
        self.df_categories.loc[self.df_categories["name"] == old_name, "name"] = new_name

    def get_categories(self):
        return self.df_categories.set_index("id").to_dict("index")

    def get_category_name(self, category_id):
        row = self.df_categories.loc[self.df_categories["id"] == category_id, "name"]
        if row.size>0:
            category_name = row.iloc[0]
        else:
            category_name = None
        return category_name

    def get_category_id(self, category_name):
        row = self.df_categories.loc[self.df_categories["name"] == category_name, "id"]
        if row.size>0:
            category_id = row.iloc[0]
        else:
            category_id = None
        return category_id

    # --------------------
    # Metadata
    # --------------------
    def set_image_metadata(self, image_id: int, key: str, value: Any):
        md = self.df_images.loc[self.df_images["id"] == image_id, "metadata"].iloc[0] or {}
        md[key] = value
        self.df_images.loc[self.df_images["id"] == image_id, "metadata"] = [md]

    def get_image_metadata(self, image_id: int) -> dict:
        return self.df_images.loc[self.df_images["id"] == image_id, "metadata"].iloc[0]

    def remove_image_metadata(self, image_id: int):
        self.df_images.loc[self.df_images["id"] != image_id]["metadata"] = {}

    def set_annotation_metadata(self, annotation_id: int, key: str, value: Any):
        md = self.df_annotations.loc[self.df_annotations["id"] == annotation_id, "metadata"].iloc[0] or {}
        md[key] = value
        self.df_annotations.loc[self.df_annotations["id"] == annotation_id, "metadata"] = [md]

    def get_annotation_metadata(self, annotation_id: int) -> dict:
        return self.df_images.loc[self.df_annotations["id"] == annotation_id, "metadata"].iloc[0]

    def remove_annotation_metadata(self, annotation_id: int):
        self.df_annotations.loc[self.df_annotations["id"] != annotation_id]["metadata"] = {}

    # --------------------
    # Plotting
    # --------------------
    def plot_image_with_annotations(self, image_id: int):
        img_row = self.get_image(image_id)
        if img_row is None:
            raise ValueError(f"Image ID {image_id} not found.")
        img_path = Path(img_row["file_name"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image file {img_path} does not exist.")

        anns = self.get_image_annotations(image_id)

        im = Image.open(img_path)
        fig, ax = plt.subplots(1)
        ax.imshow(im)

        for ann in anns:
            bbox = ann["bbox"]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            cat_name = self.df_categories.loc[self.df_categories["id"] == ann["category_id"], "name"].iloc[0]
            ax.text(bbox[0], bbox[1] - 5, cat_name, color='yellow', fontsize=12, backgroundcolor="black")
        plt.show()

    # --------------------
    # Analysis
    # --------------------
    def get_images_metadata_keys(self):
        unique_keys = {key for d in  self.df_images["metadata"] for key in d.keys()}
        return unique_keys

    def get_annotations_metadata_keys(self):
        unique_keys = {key for d in  self.df_annotations["metadata"] for key in d.keys()}
        return unique_keys

    def get_category_histogram(self) -> dict:
        """
        get histogram of category labels over all annotations

        :return: annotation_cat_hist - category histogram. dict {category_id: count}
        * Note: you can get corresponding category names using self.get_categories()
        """
        # merged = self.df_annotations.merge(self.df_categories, left_on="category_id", right_on="id", suffixes=("", "_cat"))
        annotation_cat_hist = self.df_annotations['category_id'].value_counts()
        return annotation_cat_hist.to_dict()

    def get_images_metadata_histogram(self, metadata_key: str) -> dict:
        """
        get histogram of a specific metadata field over all images
        :param metadata_key - key name string
        :return: metadata histogram. dict {value: count}
        """
        metadata_values = self.df_images["metadata"].apply(lambda d: d.get(metadata_key, None))
        return metadata_values.value_counts().sort_index().to_dict()

    def get_annotations_metadata_histogram(self, metadata_key: str) -> dict:
        """
        get histogram of a specific metadata field over all annotations
        :param metadata_key - key name string
        :return: metadata histogram. dict {value: count}
        """
        metadata_values = self.df_annotations["metadata"].apply(lambda d: d.get(metadata_key, None))
        return metadata_values.value_counts().sort_index().to_dict()

    def get_annotations_area_histogram(self, nbins=10):
        """
        get histogram of annotation area over all annotations
        :return: area histogram. dict {value: count}
        """
        # annotation_area_hist = self.df_annotations['area'].value_counts().to_dict()
        data = np.array(self.df_annotations['area'])
        hist, bin_edges = np.histogram(data, bins=nbins, range=(0, max(data)))
        return hist, bin_edges

    # ------------------- YOLO Import/Export -------------------
    def export_yolo(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        cat_mapping = dict(zip(self.df_categories["id"], self.df_categories["name"]))
        for _, row in self.df_images.iterrows():
            img_id = row["id"]
            width, height = row["width"], row["height"]
            anns = self.get_annotations_by_image(img_id)
            lines = []
            for _, ann in anns.iterrows():
                x, y, w, h = ann["bbox"]
                # convert to YOLO normalized format
                x_c = (x + w / 2) / width
                y_c = (y + h / 2) / height
                w_n = w / width
                h_n = h / height
                lines.append(f"{ann['category_id']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
            txt_path = Path(output_dir) / (Path(row["file_name"]).stem + ".txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))

    def import_yolo(self, images_dir: str, labels_dir: str):
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            image_id = self.add_image(str(img_file))
            txt_file = labels_dir / (img_file.stem + ".txt")
            if not txt_file.exists():
                continue
            width, height = self.get_image(image_id)["width"], self.get_image(image_id)["height"]
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cat_id = int(parts[0])
                    x_c, y_c, w_n, h_n = map(float, parts[1:])
                    x = x_c * width - w_n * width / 2
                    y = y_c * height - h_n * height / 2
                    w = w_n * width
                    h = h_n * height
                    self.add_annotation(image_id, cat_id, [x, y, w, h])


def _bbox_iou(box1, box2):
    """
    Compute IoU between two boxes.
    boxA, boxB = [x1, y1, x2, y2]
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute width and height of intersection
    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter_area = inter_width * inter_height

    # Compute area of both boxes
    boxA_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxB_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute IoU
    iou_value = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou_value

# --------------------
# Example usage
# --------------------
# if __name__ == "__main__":
#     manager = CocoDatasetManager()
#     img_id = manager.add_image("example.jpg")  # automatically fetch width/height
#     cat_id = manager.add_category("person")
#     ann_id = manager.add_annotation(img_id, cat_id, [10, 10, 50, 100])
#     manager.plot_image_with_annotations(img_id)
#     print(manager.analyze_by_category())
