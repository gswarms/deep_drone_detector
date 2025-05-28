import json
import os
from pathlib import Path
from PIL import Image
from typing import List, Dict, Union


class CocoDatasetManager:
    def __init__(self):
        self.data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.filename_to_image_id = {}

    def load_coco(self, json_path: Union[str, Path, List]):
        if isinstance(json_path, str) or isinstance(json_path, Path):
            json_path = [json_path]

        for jp in json_path:
            if os.path.isfile(jp):
                raise Exception('json file not found! {}'.format(jp))

            with open(jp, "r") as f:
                coco = json.load(f)

            if not self.data["categories"]:
                self.data["categories"] = coco.get("categories", [])

            for img in coco.get("images", []):
                new_img = img.copy()
                new_img["id"] = self.image_id_counter
                self.filename_to_image_id[img["file_name"]] = self.image_id_counter
                self.data["images"].append(new_img)
                self.image_id_counter += 1

            for ann in coco.get("annotations", []):
                new_ann = ann.copy()
                old_img = next((i for i in coco["images"] if i["id"] == ann["image_id"]), None)
                if old_img is None:
                    continue
                new_ann["id"] = self.annotation_id_counter
                new_ann["image_id"] = self.filename_to_image_id[old_img["file_name"]]
                self.data["annotations"].append(new_ann)
                self.annotation_id_counter += 1

        return

    def load_yolo_dataset(self, image_dir: str, label_dir: str, class_names: List[str]):
        if not self.data["categories"]:
            self.data["categories"] = [{"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(class_names)]

        for image_file in Path(image_dir).glob("*.*"):
            label_file = Path(label_dir) / f"{image_file.stem}.txt"
            if not label_file.exists():
                continue

            with Image.open(image_file) as img:
                width, height = img.size

            image_id = self.image_id_counter
            self.filename_to_image_id[image_file.name] = image_id
            self.data["images"].append({
                "id": image_id,
                "file_name": image_file.name,
                "width": width,
                "height": height
            })
            self.image_id_counter += 1

            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, w, h = map(float, parts)
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2

                    self.data["annotations"].append({
                        "id": self.annotation_id_counter,
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [x_min, y_min, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    self.annotation_id_counter += 1

    def save_yaml(self, output_yaml: Union[str, Path], dataset_root: Union[str, Path]):
        """
        Save a .yaml config file for YOLO training.
        Note: this does NOT copy or move any files.
        """
        yaml_dict = {
            "path": str(dataset_root),
            "train": "images/train",  # user should adjust paths if needed
            "val": "images/val",
            "names": {cat["id"]: cat["name"] for cat in self.data["categories"]}
        }
        with open(output_yaml, "w") as f:
            for key, val in yaml_dict.items():
                if isinstance(val, dict):
                    f.write(f"{key}:\n")
                    for k, v in val.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {val}\n")
        print(f"Saved YOLO .yaml config to {output_yaml}")

    def export_to_json(self, output_path: Union[str, Path]):
        with open(output_path, "w") as f:
            json.dump(self.data, f, indent=4)
        print(f"Exported COCO dataset to {output_path}")


if __name__ == '__main__':
    # Initialize manager
    manager = CocoDatasetManager()

    # Load multiple COCO datasets
    manager.load_multiple_coco([
        "dataset1/annotations/instances_train.json",
        "dataset2/annotations/instances_train.json"
    ])

    # Load a YOLO dataset
    manager.load_yolo_dataset(
        image_dir="yolo/images/train",
        label_dir="yolo/labels/train",
        class_names=["cat", "dog", "person"]
    )

    # Save merged dataset as COCO JSON
    manager.export_to_json("merged_coco.json")

    # Save YAML config for YOLO training
    manager.save_yaml("dataset.yaml", dataset_root="/path/to/data")
