# Dataset Utils

The dataset_utils folder is a package for handling the object detection dataset.  


## 🚀 Features

1. basic dataset management
  - load / save dataset 
  - merg datasets
  - add / remove images
  - add / remove / update annotation
  - add / remove / remap categories
  - add / remove / update image metadata
  - Export / import from YOLO format
  
2. prepare for training
  - split to train / val / test
  - balance by categories
  - balance by images metadata
  - balance by annotations metadata

3. annotate new images


## 📁 Dataset Format

We Use the [COCO dataset format](https://cocodataset.org/?utm_source=chatgpt.com#format-data)

1. dataset folder structure:
```text
dataset_root_folder/
├── annotations/
│   └── some_name.json
├── images/            # Tracking algorithms (e.g., OpenCV, DeepSORT)
│   ├── img1.png
│   ├── ...
│   └── img1000.png
```

2. annotations json file format:
```text

          {
          "info": {
            "year": 2025,
            "version": "1.0",
            "description": "Example COCO dataset",
            "contributor": "",
            "date_created": "2025-11-11"
          },
          "licenses": [
            {
              "id": 1,
              "name": "MIT",
              "url": "https://opensource.org/licenses/MIT"
            }
          ],

          "images": [
            {
              "id": 1,                    #  unique integer identifier for the image.
              "file_name": "image1.jpg",  # filename of the image relative to 'dataset_root_folder/images'
              "width": 1920,              #  image width in pixels.
              "height": 1080              #  image height in pixels.
              "license": 1080             #  *** Optionall
              "flickr_url": some url      #  *** Optionall
              "coco_url": some url        #  *** Optionall
              "date_captured": ???        #  *** Optionall
            },
            {
              "id": 2,
              "file_name": "image2.jpg",
              "width": 1280,
              "height": 720
            }
          ],

          "annotations": [
            {
              "id": 1,                       # unique annotation ID (integer).
              "image_id": 1,                 # integer — reference to the corresponding image’s `id`.
              "category_id": 1,              # integer — refers to the class (must match a category’s `id`).
              "bbox": [100, 200, 150, 300],  # bounding box in absolute pixel values: `[xtl, ytl, width, height]`.
              "area": 150 * 300,             # float — area of the object region (often bounding box area or segmentation mask area).
              "iscrowd": 0                   # integer (0 or 1) — indicates whether the annotation is for a “crowd” (group of objects) or a single instance.
              "segmentation": ???            # Optional — provides either polygon coordinates or mask encoding (RLE), for tasks needing instance segmentation.
            },
            {
              "id": 2,
              "image_id": 2,
              "category_id": 2,
              "bbox": [400, 100, 120, 200],
              "area": 120 * 200,
              "iscrowd": 0  #  important if you have group annotations (crowds). 0 or 1.
            }
          ],

          "categories": [
            {
              "id": 1,                  #  unique integer ID for the class.
              "name": "person",         # class name
              "supercategory": "human"  # (optional) broader category name for grouping classes.
            },
            {
              "id": 2,
              "name": "car",  #  class label
              "supercategory": "vehicle"  # grouping (optional)
            }
          ]
        }
        }
```
