import os
import shutil

import pytest
import tempfile
import json
from pathlib import Path
import  numpy as np
from PIL import Image
from coco_dataset_manager import CocoDatasetManager  # replace with your module name


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def empty_manager():
    return CocoDatasetManager()

@pytest.fixture
def random_image_factory():
    def _make_image(img_path, width=200, height=100, name="img.png"):
        from PIL import Image
        # Random pixels
        pixels = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(pixels, 'RGB')
        img_path = img_path / name
        img.save(img_path)
        return img_path
    return _make_image

@pytest.fixture
def sample_image(tmp_path):
    # create a dummy image file
    from PIL import Image
    img_path = tmp_path / "image.jpg"

    # Image size
    width, height = 200, 100

    # Random RGB values (0-255)
    random_pixels = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Create PIL Image from NumPy array
    im = Image.fromarray(random_pixels, 'RGB')

    # im = Image.new("RGB", (100, 200))
    im.save(img_path)
    return img_path


@pytest.fixture
def sample_category(empty_manager):
    cat_id = empty_manager.add_category("person", "human")
    return cat_id


@pytest.fixture
def sample_image_with_category(empty_manager, sample_image, sample_category):
    img_id = empty_manager.add_image(str(sample_image))
    ann_id = empty_manager.add_annotation(img_id, sample_category, [10, 20, 30, 40])
    return empty_manager, img_id, ann_id, sample_category



@pytest.fixture
def sample_dataset_with_annotations(empty_manager, random_image_factory, tmp_path):

    # set root
    dataset_root = tmp_path
    empty_manager.set_root(dataset_root)

    img1_path = random_image_factory(tmp_path, 200, 100, "tmp_img1.png")
    img1_path_relative = os.path.relpath(img1_path, empty_manager.images_folder)
    img2_path = random_image_factory(tmp_path, 200, 100, "tmp_img2.png")
    img2_path_relative = os.path.relpath(img2_path, empty_manager.images_folder)
    img3_path = random_image_factory(tmp_path, 200, 100, "tmp_img3.png")
    img3_path_relative = os.path.relpath(img3_path, empty_manager.images_folder)

    # add images
    img_id1 = empty_manager.add_image(str(img1_path), metadata={'ia':11, 'ib':12})
    img_id2 = empty_manager.add_image(str(img2_path), metadata={'ia':21, 'ib':22})
    img_id3 = empty_manager.add_image(str(img3_path), metadata={'ia':31, 'ib':32})
    # add categories
    cat_id1 = empty_manager.add_category("cat", "animal")
    cat_id2 = empty_manager.add_category("dog", "animal")

    # add annotations
    bbox1 = [10,20,30,40]
    ann_id1 = empty_manager.add_annotation(img_id1, cat_id1, bbox1, area=77, iscrowd=0, metadata={'a':1, 'b':2})
    bbox2 = [20,30,40,50]
    ann_id2 = empty_manager.add_annotation(img_id1, cat_id1, bbox2, area=78, iscrowd=1, metadata={})
    bbox3 = [30,40,50,60]
    ann_id3 = empty_manager.add_annotation(img_id1, cat_id2, bbox3, area=79, iscrowd=0, metadata={'a':10, 'b':20})
    bbox4 = [40,50,60,70]
    ann_id4 = empty_manager.add_annotation(img_id2, cat_id1, bbox4, area=80, iscrowd=1, metadata={'a':100, 'b':200})

    ret = {'dataset': empty_manager,
           'images': {img_id1: {'file_path': img1_path_relative, 'size': (200, 100), 'metadata': {'ia':11, 'ib':12}},
                      img_id2: {'file_path': img2_path_relative, 'size': (200, 100), 'metadata': {'ia':21, 'ib':22}},
                      img_id3: {'file_path': img3_path_relative, 'size': (200, 100), 'metadata': {'ia':31, 'ib':32}}},
           'categories': {cat_id1: {'name': "cat", 'supercategory': "animal"},
                          cat_id2: {'name': "dog", 'supercategory': "animal"}},
           'annotations': {ann_id1: {'img_id': img_id1, 'cat_id': cat_id1, 'bbox': bbox1, 'area': 77, 'iscrowd': 0, 'metadata': {'a': 1, 'b': 2}},
                           ann_id2: {'img_id': img_id1, 'cat_id': cat_id1, 'bbox': bbox2, 'area': 78, 'iscrowd': 1, 'metadata': {}},
                           ann_id3: {'img_id': img_id1, 'cat_id': cat_id2, 'bbox': bbox3, 'area': 79, 'iscrowd': 0, 'metadata': {'a': 10, 'b': 20}},
                           ann_id4: {'img_id': img_id2, 'cat_id': cat_id1, 'bbox': bbox4, 'area': 80, 'iscrowd': 1, 'metadata': {'a': 100, 'b': 200}}
                           }
           }
    return ret


# -----------------------------
# Tests: Images
# -----------------------------
def test_add_image(empty_manager, random_image_factory, tmp_path):

    img1_path = random_image_factory(tmp_path, 200, 100, "tmp_img1.png")

    # try to add image without setting root folder (should except)
    with pytest.raises(Exception, match="root folder must be set before adding images"):
        img_id1 = empty_manager.add_image(str(img1_path))

    # set root
    dataset_root = tmp_path
    empty_manager.set_root(dataset_root)
    img1_path_relative = os.path.relpath(img1_path, empty_manager.images_folder)

    # add image without image size
    img_id1 = empty_manager.add_image(str(img1_path))
    assert empty_manager.get_image_ids() == [img_id1]
    img_data = empty_manager.get_image(img_id1)
    assert (img_data["width"] == 200 and img_data["height"] == 100 and img_data["id"] == img_id1  and
            img_data["file_name"] == img1_path_relative and  img_data["metadata"] == {})

    # add image with wrong image size
    img2_path = random_image_factory(tmp_path, 220, 110, "tmp_img2.png")
    img2_path_relative = os.path.relpath(img2_path, empty_manager.images_folder)
    with pytest.raises(Exception):
        img_id2 = empty_manager.add_image(str(img2_path), (222, 111))

    # add image with image size
    img_id2 = empty_manager.add_image(str(img2_path), (220, 110))
    assert empty_manager.get_image_ids() == [img_id1, img_id2]
    img_data = empty_manager.get_image(img_id2)
    assert (img_data["width"] == 220 and img_data["height"] == 110 and img_data["id"] == img_id2  and
            img_data["file_name"] == img2_path_relative and  img_data["metadata"] == {})

    # add non-existing image file
    img3_path = "./non_existing_path/tmp_img2.png"
    with pytest.raises(Exception):
        img_id3 = empty_manager.add_image(str(img3_path))

    # add image that is already in the dataset
    with pytest.raises(Exception):
        img_id2 = empty_manager.add_image(str(img2_path))

    # add image with metadata
    img3_path = random_image_factory(tmp_path, 220, 110, "tmp_img3.png")
    m_data = {'a':1, 'b':2}
    img_id3 = empty_manager.add_image(str(img3_path), image_size=(220, 110), metadata=m_data)
    img3_path_relative = os.path.relpath(img3_path, empty_manager.images_folder)
    img_data = empty_manager.get_image(img_id3)
    assert (img_data["width"] == 220 and img_data["height"] == 110 and img_data["id"] == img_id3 and
            img_data["file_name"] == img3_path_relative and  img_data["metadata"] == m_data)

    # update image metadata
    m_data2 = {'a': 11, 'b': 22, 'c': 33}
    empty_manager.update_image_metadata(img_id3, m_data2)
    img_data = empty_manager.get_image(img_id3)
    assert (img_data["width"] == 220 and img_data["height"] == 110 and img_data["id"] == img_id3 and
            img_data["file_name"] == img3_path_relative and  img_data["metadata"] == m_data2)

def test_remove_image(empty_manager, random_image_factory, tmp_path):
    # TODO: remove existing image
    # TODO: remove non existing image

    img1_path = random_image_factory(tmp_path, 200, 100, "tmp_img1.png")
    img2_path = random_image_factory(tmp_path, 200, 100, "tmp_img2.png")

    # set root
    empty_manager.set_root(str(Path(img1_path).parent.parent))

    # add image
    img_id = empty_manager.add_image(str(img1_path))

    # remove image
    res = empty_manager.remove_image(img_id)
    assert res is True
    assert empty_manager.get_image(img_id) is None

    # add images
    img_id1 = empty_manager.add_image(str(img1_path))
    img_id2 = empty_manager.add_image(str(img2_path))

    # remove image
    res = empty_manager.remove_image(img_id1)
    assert res is True
    assert empty_manager.get_image(img_id1) is None
    assert empty_manager.get_image_ids() == [img_id2]

    # remove non-existent image
    res = empty_manager.remove_image(42)
    assert res is False

def test_remove_missing_images(empty_manager, random_image_factory, tmp_path):
    img1_path = random_image_factory(tmp_path, 200, 100, "tmp_img1.png")
    img2_path = random_image_factory(tmp_path, 200, 100, "tmp_img2.png")
    img3_path = random_image_factory(tmp_path, 200, 100, "tmp_img3.png")
    img4_path = random_image_factory(tmp_path, 200, 100, "tmp_img4.png")
    img5_path = random_image_factory(tmp_path, 200, 100, "tmp_img5.png")

    # set root
    empty_manager.set_root(str(Path(img1_path).parent.parent))

    # add images
    img_id1 = empty_manager.add_image(str(img1_path))
    img_id2 = empty_manager.add_image(str(img2_path))
    img_id3 = empty_manager.add_image(str(img3_path))
    img_id4 = empty_manager.add_image(str(img4_path))
    img_id5 = empty_manager.add_image(str(img5_path))

    # delete some image files
    img2_path.unlink()
    img4_path.unlink()
    img5_path.unlink()

    # remove missing images
    res = empty_manager.remove_missing_images()
    assert res == [img_id2, img_id4, img_id5]
    assert empty_manager.get_image_ids() == [img_id1, img_id3]


def test_next_image_id(empty_manager, random_image_factory, tmp_path):

    img1_path = random_image_factory(tmp_path, 200, 100, "tmp_img1.png")
    img2_path = random_image_factory(tmp_path, 200, 100, "tmp_img2.png")
    img3_path = random_image_factory(tmp_path, 200, 100, "tmp_img3.png")

    # set root
    empty_manager.set_root(str(Path(img1_path).parent.parent))

    # get next image id with no images
    next_image_id = empty_manager.get_next_image_id()
    assert next_image_id is None

    # add images
    img_id1 = empty_manager.add_image(str(img1_path))
    img_id2 = empty_manager.add_image(str(img2_path))
    img_id3 = empty_manager.add_image(str(img3_path))

    # get next image id first time
    next_image_id = empty_manager.get_next_image_id()
    assert next_image_id is img_id1
    next_image_id = empty_manager.get_next_image_id()  # shouldnt increase because we didn't get image
    assert next_image_id is img_id1

    # get next image id
    img_data = empty_manager.get_image(img_id2)
    next_image_id = empty_manager.get_next_image_id()
    assert next_image_id is img_id3
    next_image_id = empty_manager.get_next_image_id()   # shouldnt increase because we didn't get image
    assert next_image_id is img_id3
    img_data = empty_manager.get_image(img_id1)
    next_image_id = empty_manager.get_next_image_id()
    assert next_image_id is img_id2

    # after last image
    img_data = empty_manager.get_image(img_id3)
    next_image_id = empty_manager.get_next_image_id()
    assert next_image_id is None

    # get next image id after required id
    next_image_id = empty_manager.get_next_image_id(img_id2)
    assert next_image_id is img_id3
    next_image_id = empty_manager.get_next_image_id(img_id1)
    assert next_image_id is img_id2
    next_image_id = empty_manager.get_next_image_id(img_id3)
    assert next_image_id is None
    next_image_id = empty_manager.get_next_image_id(10)
    assert next_image_id is None
    next_image_id = empty_manager.get_next_image_id(-5)
    assert next_image_id is img_id1


# -----------------------------
# Tests: Categories
# -----------------------------
def test_add_get_category(empty_manager):

    cat_id = empty_manager.add_category("cat", "animal")
    assert cat_id in empty_manager.df_categories["id"].values

    dog_id = empty_manager.add_category("dog", "animal")
    assert dog_id in empty_manager.df_categories["id"].values

    cat = empty_manager.get_categories()
    assert cat == {1: {'name': 'cat', 'supercategory': 'animal'},
                   2: {'name': 'dog', 'supercategory': 'animal'}}

    # add existing category
    with pytest.raises(ValueError):
        cat_id = empty_manager.add_category("cat", "animal")


def test_update_category(empty_manager):

    cat_id = empty_manager.add_category("cat", "animal")
    assert cat_id in empty_manager.df_categories["id"].values

    dog_id = empty_manager.add_category("dog", "animal")
    assert dog_id in empty_manager.df_categories["id"].values

    # update category
    empty_manager.update_category(dog_id, "puppy", "cute_animal")
    cat = empty_manager.get_categories()
    assert cat == {1: {'name': 'cat', 'supercategory': 'animal'},
                   2: {'name': 'puppy', 'supercategory': 'cute_animal'}}

    # update non-existing
    with pytest.raises(ValueError):
        empty_manager.update_category(7, "puppy", "cute_animal")


def test_remove_category(empty_manager):

    cat_id = empty_manager.add_category("cat", "animal")
    assert cat_id in empty_manager.df_categories["id"].values

    dog_id = empty_manager.add_category("dog", "animal")
    assert dog_id in empty_manager.df_categories["id"].values

    cat = empty_manager.get_categories()
    assert cat == {1: {'name': 'cat', 'supercategory': 'animal'},
                   2: {'name': 'dog', 'supercategory': 'animal'}}

    # remove category
    res = empty_manager.remove_category(cat_id)
    assert res is True
    assert cat_id not in empty_manager.df_categories["id"].values
    cat = empty_manager.get_categories()
    assert cat == {2: {'name': 'dog', 'supercategory': 'animal'}}

    res = empty_manager.remove_category(7)
    assert res is False


def test_remap_category_name(empty_manager):
    cat_id1 = empty_manager.add_category("cat", "animal")
    cat_id2 = empty_manager.add_category("dog", "animal")

    empty_manager.remap_category_name("dog", "lavrador")
    assert "lavrador" in empty_manager.df_categories["name"].values
    assert "dog" not in empty_manager.df_categories["name"].values
    cat = empty_manager.get_categories()
    assert cat == {1: {'name': 'cat', 'supercategory': 'animal'},
                   2: {'name': 'lavrador', 'supercategory': 'animal'}}


def test_get_category_name(empty_manager):
    cat_id1 = empty_manager.add_category("cat", "animal")
    cat_id2 = empty_manager.add_category("dog", "animal")


    c1 = empty_manager.get_category_name(cat_id1)
    assert c1=="cat"
    c2 = empty_manager.get_category_name(cat_id2)
    assert c2=="dog"
    c3 = empty_manager.get_category_name(7)
    assert c3 is None

    id1 = empty_manager.get_category_id("cat")
    assert id1 == cat_id1
    id2 = empty_manager.get_category_id("dog")
    assert id2 == cat_id2
    id3 = empty_manager.get_category_name("camel")
    assert id3 is None


# -----------------------------
# Tests: Annotations
# -----------------------------
def test_add_annotation(sample_dataset_with_annotations):

    dataset_manager = sample_dataset_with_annotations['dataset']
    annotations = sample_dataset_with_annotations['annotations']

    for ann_id in annotations:
        ann_data_ref = annotations[ann_id]
        ann_data = dataset_manager.get_annotation(ann_id)
        assert ann_data == {'id': ann_id, 'image_id': ann_data_ref['img_id'], 'category_id': ann_data_ref['cat_id'],
                            'bbox': ann_data_ref['bbox'], 'area': ann_data_ref['area'], 'segmentation': [],
                            'iscrowd': ann_data_ref['iscrowd'], 'metadata': ann_data_ref['metadata']}


def test_get_annotation(sample_dataset_with_annotations):

    dataset_manager = sample_dataset_with_annotations['dataset']
    annotations = sample_dataset_with_annotations['annotations']

    # get valid annotations
    for ann_id in annotations.keys():
        ann_data_ref = annotations[ann_id]
        ann_data = dataset_manager.get_annotation(ann_id)
        assert ann_data == {'id': ann_id, 'image_id': ann_data_ref['img_id'], 'category_id': ann_data_ref['cat_id'],
                            'bbox': ann_data_ref['bbox'], 'area': ann_data_ref['area'], 'segmentation': [],
                            'iscrowd': ann_data_ref['iscrowd'], 'metadata': ann_data_ref['metadata']}

    # get non existing valid annotation
    ann_data2 = dataset_manager.get_annotation(42)
    assert ann_data2 is None

    # get annotations by image id
    img_id = 1
    anns = dataset_manager.get_image_annotations(img_id)
    anns_ids = [x['id'] for x in anns]
    assert len(anns) == 3 and anns_ids == [1, 2, 3]

    anns = dataset_manager.get_image_annotations(7)  # non existent image
    assert len(anns) == 0


def test_update_annotation(sample_dataset_with_annotations):

    dataset_manager = sample_dataset_with_annotations['dataset']
    annotations = sample_dataset_with_annotations['annotations']

    # update annotations
    ann_id1 = 1
    dataset_manager.update_annotation(ann_id1, bbox=[11,21,31,41], category_id=2, area=777, iscrowd=1, metadata={'a':11, 'b':21})
    ann_data1 = dataset_manager.get_annotation(ann_id1)
    assert ann_data1 == {'id': ann_id1, 'image_id': annotations[ann_id1]['img_id'], 'category_id': 2, 'bbox': [11,21,31,41],
                         'segmentation': [], 'area': 777, 'iscrowd': 1, 'metadata': {'a': 11, 'b': 21}}

    ann_id2 = 2
    ann_data_ref = annotations[ann_id2]
    ann_data = dataset_manager.get_annotation(ann_id2)
    assert ann_data == {'id': ann_id2, 'image_id': ann_data_ref['img_id'], 'category_id': ann_data_ref['cat_id'],
                        'bbox': ann_data_ref['bbox'], 'area': ann_data_ref['area'], 'segmentation': [],
                        'iscrowd': ann_data_ref['iscrowd'], 'metadata': ann_data_ref['metadata']}

    # update non-existing annotation
    with pytest.raises(Exception):
        dataset_manager.update_annotation(77, bbox=[11, 21, 31, 41], category_id=2, area=777, iscrowd=1,
                                        metadata={'a': 11, 'b': 21})

    # update annotation with non-existing field
    with pytest.raises(Exception):
        dataset_manager.update_annotation(ann_id1, bbox=[11, 21, 31, 41], category_id=2, religion='Flying_Spaghetti_Monster')


def test_remove_annotation(sample_dataset_with_annotations):

    dataset_manager = sample_dataset_with_annotations['dataset']
    annotations = sample_dataset_with_annotations['annotations']

    # remove annotations
    ann_id2 = 2
    dataset_manager.remove_annotation(ann_id2)
    assert dataset_manager.get_annotation(ann_id2) is None

    # remove annotations by removing category
    cat_id2 = 2
    dataset_manager.remove_category(cat_id2)
    assert dataset_manager.get_annotation(3) is None

    ann_id1 = 1
    ann_data_ref = annotations[ann_id1]
    ann_data = dataset_manager.get_annotation(ann_id1)
    assert ann_data == {'id': ann_id1, 'image_id': ann_data_ref['img_id'], 'category_id': ann_data_ref['cat_id'],
                        'bbox': ann_data_ref['bbox'], 'area': ann_data_ref['area'], 'segmentation': [],
                        'iscrowd': ann_data_ref['iscrowd'], 'metadata': ann_data_ref['metadata']}

    # remove annotations by removing image
    image_id = 2
    dataset_manager.remove_image(image_id)
    assert dataset_manager.get_annotation(4) is None


# -----------------------------
# Tests: Save / Load JSON
# -----------------------------
def test_save_load_coco(sample_dataset_with_annotations, tmp_path):
    # TODO: test save with / without copy images

    dataset_manager = sample_dataset_with_annotations['dataset']
    img_ref = sample_dataset_with_annotations['images']
    ann_ref = sample_dataset_with_annotations['annotations']
    cat_ref = sample_dataset_with_annotations['categories']

    # Save dataset
    json_path = tmp_path / "annotations" / "dataset.json"
    dataset_manager.save_coco(dataset_root_folder=tmp_path, json_file_name="dataset.json", copy_images=True)

    # Load into a new manager
    new_manager = CocoDatasetManager()
    new_manager.load_coco(json_path, verify_image_files=True)

    for img_id in img_ref.keys():
        img_data_ref = img_ref[img_id]
        img_data = dataset_manager.get_image(img_id)
        assert img_data == {'id': img_id, 'file_name': img_data_ref['file_path'],
                            'width': img_data_ref['size'][0], 'height': img_data_ref['size'][1],
                            'metadata': img_data_ref['metadata']}

    cat_data = dataset_manager.get_categories()
    assert cat_data.keys() == cat_ref.keys()
    for cat_id in cat_ref.keys():
        assert cat_data[cat_id] == {'name': cat_ref[cat_id]['name'], 'supercategory': cat_ref[cat_id]['supercategory']}

    for ann_id in ann_ref.keys():
        ann_data_ref = ann_ref[ann_id]
        ann_data = dataset_manager.get_annotation(ann_id)
        assert ann_data == {'id': ann_id, 'image_id': ann_data_ref['img_id'], 'category_id': ann_data_ref['cat_id'],
                            'bbox': ann_data_ref['bbox'], 'area': ann_data_ref['area'], 'segmentation': [],
                            'iscrowd': ann_data_ref['iscrowd'], 'metadata': ann_data_ref['metadata']}

    # Load dataset with missing image files
    im = dataset_manager.get_image(2)
    im_path = (dataset_manager.images_folder / Path(im['file_name'])).resolve()
    os.remove(im_path)
    with pytest.raises(Exception):
        new_manager.load_coco(json_path, verify_image_files=True)

    # Load non existing json
    json_path = tmp_path / "dataset_non_existing.json"
    with pytest.raises(Exception):
        new_manager.load_coco(json_path, verify_image_files=False)


# -----------------------------
# Tests: Merge datasets
# -----------------------------
def test_merge_datasets(sample_dataset_with_annotations, random_image_factory, tmp_path):

    # TODO: merge datasets with no duplicates
    # TODO: merge datasets with image path duplicates
    # TODO: merge datasets with image hash duplicates
    # TODO: merge datasets with image path and hash duplicates

    # TODO: merge datasets with annotatino duplicates - with / without overlaps

    # TODO: test category remap

    # ----------------------- test category renaming --------------------------
    # create manager
    dataset_manager = sample_dataset_with_annotations['dataset']
    img_ref = sample_dataset_with_annotations['images']
    ann_ref = sample_dataset_with_annotations['annotations']
    cat_ref = sample_dataset_with_annotations['categories']

    # create another manager
    other_manager = CocoDatasetManager()
    # set root
    other_manager.set_root(tmp_path / 'other')

    img1_path = random_image_factory(tmp_path, 200, 100, "tmp_img21.png")
    img1_path_relative = os.path.relpath(img1_path, other_manager.images_folder)
    img2_path = random_image_factory(tmp_path, 200, 100, "tmp_img22.png")
    img2_path_relative = os.path.relpath(img2_path, other_manager.images_folder)

    # add images
    img_id1 = other_manager.add_image(str(img1_path), metadata={'ia':211, 'ib':212})
    img_id2 = other_manager.add_image(str(img2_path), metadata={'ia':221, 'ib':222})
    # add categories
    cat_id1 = other_manager.add_category("dog", "animal")
    cat_id2 = other_manager.add_category("donkey", "animal")
    cat_id3 = other_manager.add_category("cat", "animal")
    # add annotations
    bbox1 = [210,220,230,240]
    ann_id1 = other_manager.add_annotation(img_id1, cat_id1, bbox1, area=81, iscrowd=0, metadata={'a':1000, 'b':2000})
    bbox2 = [220,230,240,250]
    ann_id2 = other_manager.add_annotation(img_id1, cat_id1, bbox2, area=82, iscrowd=1, metadata={})
    bbox3 = [230,240,250,260]
    ann_id3 = other_manager.add_annotation(img_id1, cat_id2, bbox3, area=83, iscrowd=0, metadata={'a':1001, 'b':2001})
    bbox4 = [240,250,260,270]
    ann_id4 = other_manager.add_annotation(img_id2, cat_id3, bbox4, area=84, iscrowd=1, metadata={'a':1002, 'b':2002})

    # merge
    dataset_manager.merge_dataset(other_manager)

    # test results
    img_ref[4] = {'file_path': str(Path(img1_path_relative).name),  # after this image is copied to images
                  'size': (200, 100),
                  'metadata': {'ia':211, 'ib':212}}
    img_ref[5] = {'file_path': str(Path(img2_path_relative).name),  # after this image is copied to images
                  'size': (200, 100),
                  'metadata': {'ia':221, 'ib':222}}
    for img_id in img_ref.keys():
        img_data_ref = img_ref[img_id]
        img_data = dataset_manager.get_image(img_id)
        assert img_data == {'id': img_id, 'file_name': img_data_ref['file_path'],
                            'width': img_data_ref['size'][0], 'height': img_data_ref['size'][1],
                            'metadata': img_data_ref['metadata']}

    cat_ref[3] = {'name': 'donkey', 'supercategory': 'animal'}
    cat_data = dataset_manager.get_categories()
    assert cat_data == cat_ref

    ann_ref[5] = {'img_id': 4, 'cat_id': 2, 'bbox': bbox1, 'area': 81, 'iscrowd': 0, 'metadata': {'a': 1000, 'b': 2000}}
    ann_ref[6] = {'img_id': 4, 'cat_id': 2, 'bbox': bbox2, 'area': 82, 'iscrowd': 1, 'metadata': {}}
    ann_ref[7] = {'img_id': 4, 'cat_id': 3, 'bbox': bbox3, 'area': 83, 'iscrowd': 0, 'metadata': {'a': 1001, 'b': 2001}}
    ann_ref[8] = {'img_id': 5, 'cat_id': 1, 'bbox': bbox4, 'area': 84, 'iscrowd': 1, 'metadata': {'a': 1002, 'b': 2002}}
    for ann_id in ann_ref.keys():
        ann_data_ref = ann_ref[ann_id]
        ann_data = dataset_manager.get_annotation(ann_id)
        assert ann_data == {'id': ann_id, 'image_id': ann_data_ref['img_id'], 'category_id': ann_data_ref['cat_id'],
                            'bbox': ann_data_ref['bbox'], 'area': ann_data_ref['area'], 'segmentation': [],
                            'iscrowd': ann_data_ref['iscrowd'], 'metadata': ann_data_ref['metadata']}

    # ----------------------- test duplicate images --------------------------
    other_manager = CocoDatasetManager()
    # set root
    other_manager.set_root(tmp_path / 'other2')
    # add images
    img1_path = random_image_factory(tmp_path, 200, 100, "tmp_img31.png")
    img1_path_relative = os.path.relpath(img1_path, other_manager.images_folder)

    img2_path_relative = dataset_manager.get_image(1)['file_name']  # use an image already existing in original dataset
    img2_path = (dataset_manager.images_folder / Path(img2_path_relative)).resolve()

    img3_path_relative = dataset_manager.get_image(2)['file_name']  # hash duplicate image
    img3_orig_path = (dataset_manager.images_folder / Path(img3_path_relative)).resolve()
    img3_path = (other_manager.images_folder / Path("tmp_img33.png")).resolve()
    img3_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img3_orig_path, img3_path)

    img_id1 = other_manager.add_image(str(img1_path), metadata={'ia':311, 'ib':312})
    img_id2 = other_manager.add_image(str(img2_path), metadata={'ia':321, 'ib':322})
    img_id3 = other_manager.add_image(str(img3_path), metadata={'ia':331, 'ib':332})
    # add categories
    cat_id1 = other_manager.add_category("cat", "animal")
    cat_id2 = other_manager.add_category("dog", "animal")
    # add annotations
    bbox1 = [222,333,13,14]
    ann_id1 = other_manager.add_annotation(img_id1, cat_id1, bbox1, area=91, iscrowd=0, metadata={'a':345, 'b':2123})
    bbox2 = [11,21,30,40]  # annotation on an existing image id. this overlaps with an existing annotation bbox! Therefore should be discarded in merge.
    ann_id2 = other_manager.add_annotation(img_id2, cat_id2, bbox2, area=92, iscrowd=1, metadata={})
    bbox3 = [100,90,10,20] # annotation on an existing image id.
    ann_id3 = other_manager.add_annotation(img_id2, cat_id2, bbox3, area=93, iscrowd=1, metadata={})
    bbox4 = [1,2,10,20] # annotation on an existing image id.
    ann_id4 = other_manager.add_annotation(img_id3, cat_id2, bbox4, area=94, iscrowd=1, metadata={})

    # merge
    dataset_manager.merge_dataset(other_manager, merge_hash_duplicates=True)

    # test results

    img_ref[6] = {'file_path': str(Path(img1_path_relative).name),  # after this image is copied to images
                  'size': (200, 100),
                  'metadata': {'ia':311, 'ib':312}}
    assert len(dataset_manager.df_images)==6
    for img_id in img_ref.keys():
        img_data_ref = img_ref[img_id]
        img_data = dataset_manager.get_image(img_id)
        assert img_data == {'id': img_id, 'file_name': img_data_ref['file_path'],
                            'width': img_data_ref['size'][0], 'height': img_data_ref['size'][1],
                            'metadata': img_data_ref['metadata']}

    ann_ref[9] = {'img_id': 6, 'cat_id': 1, 'bbox': bbox1, 'area': 91, 'iscrowd': 0, 'metadata': {'a': 345, 'b': 2123}}
    ann_ref[10] = {'img_id': 1, 'cat_id': 2, 'bbox': bbox3, 'area': 93, 'iscrowd': 1, 'metadata': {}}
    ann_ref[11] = {'img_id': 2, 'cat_id': 2, 'bbox': bbox4, 'area': 94, 'iscrowd': 1, 'metadata': {}}
    for ann_id in ann_ref.keys():
        ann_data_ref = ann_ref[ann_id]
        ann_data = dataset_manager.get_annotation(ann_id)
        assert ann_data == {'id': ann_id, 'image_id': ann_data_ref['img_id'], 'category_id': ann_data_ref['cat_id'],
                            'bbox': ann_data_ref['bbox'], 'area': ann_data_ref['area'], 'segmentation': [],
                            'iscrowd': ann_data_ref['iscrowd'], 'metadata': ann_data_ref['metadata']}


# -----------------------------
# Tests: Analysis
# -----------------------------

# def test_analyze_by_category(sample_image_with_category):
#     manager, img_id, ann_id, cat_id = sample_image_with_category
#     analysis = manager.analyze_by_category()
#     assert analysis["count"].iloc[0] == 1

#----------------------------------------------- tested upto here -------------------------------------------------------------

def test_analyze_metadata(random_image_factory, tmp_path):

    dataset_manager = CocoDatasetManager()

    # set root
    dataset_root = tmp_path
    dataset_manager.set_root(dataset_root)

    img1_path = random_image_factory(tmp_path, 200, 100, "tmp_img1.png")
    img2_path = random_image_factory(tmp_path, 200, 100, "tmp_img2.png")
    img3_path = random_image_factory(tmp_path, 200, 100, "tmp_img3.png")
    img4_path = random_image_factory(tmp_path, 200, 100, "tmp_img4.png")
    img5_path = random_image_factory(tmp_path, 200, 100, "tmp_img5.png")
    img6_path = random_image_factory(tmp_path, 200, 100, "tmp_img6.png")

    # add images
    img_id1 = dataset_manager.add_image(str(img1_path), metadata={'a1':11, 'b1':12})
    img_id2 = dataset_manager.add_image(str(img2_path), metadata={'a2':21, 'b1':12})
    img_id3 = dataset_manager.add_image(str(img3_path), metadata={'a3':31, 'b1':12})
    img_id4 = dataset_manager.add_image(str(img4_path), metadata={'a1':11, 'b1':42})
    img_id5 = dataset_manager.add_image(str(img5_path), metadata={'b1':42, 'a1':52})
    img_id6 = dataset_manager.add_image(str(img6_path), metadata={'a2':61, 'c6':62})

    # add categories
    cat_id1 = dataset_manager.add_category("cat", "animal")
    cat_id2 = dataset_manager.add_category("dog", "animal")

    # add annotations
    bbox1 = [10,20,10,12]
    ann_id1 = dataset_manager.add_annotation(img_id1, cat_id1, bbox1, area=None, iscrowd=0, metadata={'a1':5, 'b1':22})
    bbox2 = [20,30,10,12]
    ann_id2 = dataset_manager.add_annotation(img_id1, cat_id1, bbox2, area=None, iscrowd=1, metadata={})
    bbox3 = [30,40,20,15]
    ann_id3 = dataset_manager.add_annotation(img_id1, cat_id2, bbox3, area=None, iscrowd=0, metadata={'a1':5, 'b1':22})
    bbox4 = [40,50,20,15]
    ann_id4 = dataset_manager.add_annotation(img_id2, cat_id1, bbox4, area=None, iscrowd=1, metadata={'a1':5, 'b1':-1})
    bbox5 = [40,50,21,15]
    ann_id5 = dataset_manager.add_annotation(img_id2, cat_id1, bbox5, area=None, iscrowd=1, metadata={'a1':6, 'b2':200})
    bbox6 = [40,50,22,15]
    ann_id6 = dataset_manager.add_annotation(img_id3, cat_id1, bbox6, area=None, iscrowd=1, metadata={'a1':6, 'b2':200})
    bbox7 = [40,50,30,20]
    ann_id7 = dataset_manager.add_annotation(img_id4, cat_id1, bbox7, area=None, iscrowd=1, metadata={'a1':7, 'b3':200})
    bbox8 = [40,50,30,20]
    ann_id8 = dataset_manager.add_annotation(img_id5, cat_id1, bbox8, area=None, iscrowd=1, metadata={'a2':100, 'b4':200})
    bbox9 = [40,50,30,21]
    ann_id9 = dataset_manager.add_annotation(img_id6, cat_id2, bbox9, area=None, iscrowd=1, metadata={'a2':100, 'b5':200})
    bbox10 = [40,50,40,40]
    ann_id10 = dataset_manager.add_annotation(img_id2, cat_id2, bbox10, area=None, iscrowd=1, metadata={'a2':100, 'b6':200})
    bbox11 = [40,50,50,50]
    ann_id11 = dataset_manager.add_annotation(img_id3, cat_id1, bbox11, area=None, iscrowd=1, metadata={'a3':100, 'b7':200})

    cat_count = dataset_manager.get_category_histogram()
    assert cat_count == {cat_id1: 8, cat_id2: 3}

    mkeys = dataset_manager.get_images_metadata_keys()
    assert sorted(mkeys) == sorted(['a1', 'a2', 'a3', 'b1', 'c6'])

    mkeys = dataset_manager.get_annotations_metadata_keys()
    assert sorted(mkeys) == sorted(['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])

    h, bin_edges = dataset_manager.get_annotations_area_histogram(nbins=15)
    assert all(h == np.array([2, 4, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])) and all(bin_edges == np.linspace(0,2500, 16))

    h = dataset_manager.get_annotations_metadata_histogram('a1')
    assert h == {5:3, 6:2, 7:1}
    h = dataset_manager.get_annotations_metadata_histogram('b1')
    assert h == {22:2, -1:1}

    h = dataset_manager.get_images_metadata_histogram('b1')
    assert h == {12:3, 42:2}


# -----------------------------
# Optional: Plotting (just ensure no exceptions)
# -----------------------------
# def test_plot_image_with_annotations(sample_image_with_category):
#     manager, img_id, ann_id, cat_id = sample_image_with_category
#     try:
#         manager.plot_image_with_annotations(img_id)
#     except FileNotFoundError:
#         # ignore if the dummy image doesn't exist
#         pass


# -----------------------------
# Run pytest from main
# -----------------------------
if __name__ == "__main__":
    import sys
    import pytest

    # Run pytest programmatically
    # Exit with pytest's exit code so it can be used in CI
    sys.exit(pytest.main([__file__, "-v", "-x"]))
    # sys.exit(pytest.main([__file__, "-v", "-x", "--tb=short"]))
