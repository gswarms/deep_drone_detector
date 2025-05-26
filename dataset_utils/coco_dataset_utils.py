import json
import os
from copy import deepcopy
import glob

def merge_coco_datasets(annotation_paths, output_path):
    """
    Merges multiple COCO-format annotation files into one.

    Parameters:
        annotation_paths (list of str): Paths to COCO JSON annotation files.
        output_path (str): Path to save the merged JSON.

    Returns:
        dict: Merged COCO annotations.
    """
    merged = {
        "images": [],
        "annotations": [],
        "categories": None
    }

    image_id_offset = 0
    annotation_id_offset = 0
    category_mapping = {}
    category_name_to_id = {}

    for i, path in enumerate(annotation_paths):
        with open(path, 'r') as f:
            coco = json.load(f)

        if i == 0:
            # Use first dataset’s categories as base
            merged["categories"] = deepcopy(coco["categories"])
            category_name_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}
            category_mapping = {cat["id"]: cat["id"] for cat in coco["categories"]}
        else:
            # Ensure categories match by name
            for cat in coco["categories"]:
                name = cat["name"]
                if name not in category_name_to_id:
                    new_id = max(category_name_to_id.values()) + 1
                    category_name_to_id[name] = new_id
                    merged["categories"].append({
                        "id": new_id,
                        "name": name,
                        "supercategory": cat.get("supercategory", "")
                    })
                category_mapping[cat["id"]] = category_name_to_id[name]

        # Remap image and annotation IDs
        for img in coco["images"]:
            old_id = img["id"]
            img["id"] += image_id_offset
            merged["images"].append(img)

        for ann in coco["annotations"]:
            ann["id"] += annotation_id_offset
            ann["image_id"] += image_id_offset
            ann["category_id"] = category_mapping[ann["category_id"]]
            merged["annotations"].append(ann)

        # Update offsets
        image_id_offset = max([img["id"] for img in merged["images"]]) + 1
        annotation_id_offset = max([ann["id"] for ann in merged["annotations"]]) + 1

    # Save merged file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as out_file:
        json.dump(merged, out_file)

    return merged


if __name__ == '__main__':

    # Example usage:
    yolo_base_dir = '/home/roee/Downloads/Drone Dataset.v2i.yolov8/dataset_baseline'
    annotation_files = glob.glob(os.path.join(yolo_base_dir,'*','*','*.json'))

    merged_output = os.path.join(yolo_base_dir,'merged_coco_dataset/test.json')

    merge_coco_datasets(annotation_files, merged_output)
    print('Done')

