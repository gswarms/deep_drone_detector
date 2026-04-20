import os
from pathlib import Path
import cv2
import yaml
import coco_dataset_manager



def add_images_common_metadata(dataset, metadata):
    """
    add the same metadata to all images in a dataset

    :param dataset: coco dataset manager
    :param metadata: metadata to add to all images - dict
    :return:
    """

    # add metadata to all images
    count = 0
    img_ids = dataset.get_image_ids()
    for im_id in img_ids:
        dataset.update_image_metadata(im_id, metadata)
        count = count + 1

    print('updated metadata to {} images'.format(count))
    print('new metadata:')
    print(metadata)


def add_annotations_common_metadata(dataset, category, metadata):
    """
    add the same metadata to all annotations of a specific category in a dataset

    :param dataset: coco manager
    :param output_json_file_name: output coco dataset json file name (not path)
    :param category: annotations category - str
                     if None - add for all categories
    :param metadata: metadata to add - dict
    :return:
    """

    # add metadata to all annotations
    annotation_ids = dataset.df_annotations['id']
    count = 0
    for ann_id in annotation_ids:
        ann_data = dataset.get_annotation(ann_id)
        if category is None or ann_data['category'] == category:
            dataset.update_annotation(ann_id, metadata=metadata)
            count = count + 1

    print('updated metadata to {} annotations'.format(count))
    print('new metadata:')
    print(metadata)

def fix_annotations_bbox(dataset):
    """
    fix annotations bbox to be inside the image borders
    :param dataset:
    :return:
    """
    # add metadata to all annotations
    annotation_ids = dataset.df_annotations['id']
    for ann_id in annotation_ids:
        ann_data = dataset.get_annotation(ann_id)

        image_data = dataset.get_image(ann_data['image_id'])

        x1 = ann_data['bbox'][0]
        y1 = ann_data['bbox'][1]
        x2 = x1 + ann_data['bbox'][2]
        y2 = y1 + ann_data['bbox'][3]

        x1 = min(max(0, x1), image_data['width']-1)
        y1 = min(max(0, y1), image_data['height']-1)
        x2 = min(max(0, x2), image_data['width']-1)
        y2 = min(max(0, y2), image_data['height']-1)

        dataset.update_annotation(ann_id, bbox=[x1, y1, x2-x1, y2-y1])

def images_bgr2rgb(dataset):
    """
    convert all images from bgr to rgb
    :param dataset:
    :return:
    """

    # add metadata to all images
    count = 0
    img_ids = dataset.get_image_ids()
    for im_id in img_ids:
        img_data = dataset.get_image(im_id)
        img_path = dataset.images_folder / img_data['file_name']
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_rgb)
        count = count + 1
    print('converted {} images'.format(count))

def _dataset_cat_ids_remap():
    """
    remap category / images / annotations ids to 0,1,2,...
    """

    base_folder = Path('/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211')

    exp_folders = []
    # exp_folders.append(base_folder / '20251214_reshafim')
    # exp_folders.append(base_folder / '20251208_reshafim')
    # exp_folders.append(base_folder / '20251204_kfar_galim')
    # exp_folders.append(base_folder / '20251126_kfar_galim')
    # exp_folders.append(base_folder / '20251027_kfar_galim')
    # exp_folders.append(base_folder / '20250917_lehavim')
    # exp_folders.append(base_folder / '20250918_lehavim')
    # exp_folders.append(base_folder / '20250608_kfar_masarik')
    # exp_folders.append(base_folder / '20250421_hadera')
    exp_folders.append(base_folder / 'merged_dataset_raw')


    discard_exp = []

    for exp_folder in exp_folders:
        for scene_folder in exp_folder.glob("**/annotations"):
            if scene_folder.parent.name in discard_exp:
                continue

            # init dataset
            scene_json = scene_folder/'coco_dataset.json'
            dataset = coco_dataset_manager.CocoDatasetManager()
            dataset.load_coco(scene_json)

            # remap ids to 0...n-1
            cat = dataset.get_categories()
            cat_ids = cat.keys()
            cat_ids_remap = {}
            for i, cid in enumerate(cat_ids):
                cat_ids_remap[cid] = i
                dataset.df_annotations["category_id"] = dataset.df_annotations["category_id"].replace(cid, i)
                dataset.df_categories["id"] = dataset.df_categories["id"].replace(cid, i)

            dataset.save_coco(dataset_root_folder=None, json_file_name=None, copy_images=False, overwrite=True)

            print('scenario: {}'.format(scene_folder))
            print('replace cat id: {}'.format(cat_ids_remap))


def load_scenario_metadata(metadata_yaml_file):
    """
    load images / annotations metadata from yaml file

    yaml file structure:
    - images:
        - param1: value1
        - param2: value2
        - ...
    - annotation:
        - param1: value1
        - param2: value2
        - ...

    :param metadata_yaml_file: path to metadata yaml file
    :return:
    """

    ret = False
    data = None
    if os.path.isfile(metadata_yaml_file):
        try:
            with open(metadata_yaml_file, "r") as f:
                data = yaml.safe_load(f)

            if data is not None:
                ret = True

        except Exception as e:
            print("Failed to read YAML:", e)
    else:
        print("YAML file: {} not found!".format(metadata_yaml_file))

    return ret, data

if __name__ == '__main__':

    # --------------------------- remap ids ------------------------------
    # _dataset_cat_ids_remap()
    # print('Done')

    # --------------------------- remove missing images ------------------------------
    # scene_json = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250608_kfar_masarik/2025-06-08_19-25-35/camera_2025_6_8-16_25_38_extracted/annotations/coco_dataset.json'
    # scene_dataset = coco_dataset_manager.CocoDatasetManager()
    # scene_dataset.load_coco(scene_json, verify_image_files=False)
    # removed_image_ids = scene_dataset.remove_missing_images()
    # print('removed {} missing images!'.format(len(removed_image_ids)))
    # scene_dataset.save_coco(overwrite=True)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5 vis dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # ------------------ hadera 21.04.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20250421_hadera'
    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_10-57-38/camera_2025_4_21-7_59_8_extracted')  # no target (almost)
    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_10-59-30/camera_2025_4_21-7_59_41_extracted')  # OK
    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_11-11-25/camera_2025_4_21-8_12_28_extracted')  # OK
    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_11-13-03/camera_2025_4_21-8_13_40_extracted')  # OK
    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_11-24-25/camera_2025_4_21-8_25_32_extracted')  # passes far - OK
    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_11-26-29/camera_2025_4_21-8_26_42_extracted')  # passes far - OK
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ kfar_masarik 08.06.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20250608_kfar_masarik'
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-04-53/camera_2025_6_8-15_4_56_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-29-51/camera_2025_6_8-15_29_54_extracted')  # passes farK
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-51-15/camera_2025_6_8-15_51_18_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-52-56/camera_2025_6_8-15_52_58_extracted')  # passes far
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-53-46/camera_2025_6_8-15_53_49_extracted')  # passes far
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-59-46/camera_2025_6_8-15_59_49_extracted')  # passes far
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_19-00-25/camera_2025_6_8-16_0_28_extracted')  # passes far
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_19-08-34/camera_2025_6_8-16_8_48_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_19-09-33/camera_2025_6_8-16_9_38_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_19-25-35/camera_2025_6_8-16_25_38_extracted')  # passes far
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ lehavim 17.09.2025 - pre test ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20250917_lehavim'
    # dataset_root_folder = os.path.join(exp_folder, 'hb003/20250915_1028_24/camera_20250915_1028_extracted')  # no target - prep day
    # dataset_root_folder = os.path.join(exp_folder, 'hb003/20250917_1747_34/camera_20250917_1747_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ lehavim 18.09.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20250918_lehavim'
    # dataset_root_folder = os.path.join(exp_folder, 'hb002/20250915_1019_36/camera_20250915_1019_extracted')  # no target - prep day
    # dataset_root_folder = os.path.join(exp_folder, 'hb002/20250918_1040_02/camera_20250918_1040_extracted')  # no target
    # dataset_root_folder = os.path.join(exp_folder, 'hb002/20250918_1041_26/camera_20250918_1041_extracted')  # no target
    # dataset_root_folder = os.path.join(exp_folder, 'hb002/20250918_1044_49/camera_20250918_1044_extracted')
    # dataset_root_folder = os.path.join(exp_folder, 'hb004/20250918_1451_40/camera_20250918_1451_extracted')
    # dataset_root_folder = os.path.join(exp_folder, 'hb004/20250918_1453_47/camera_20250918_1453_extracted')
    # dataset_root_folder = os.path.join(exp_folder, 'hb004/20250918_1456_00/camera_20250918_1456_extracted')
    # dataset_root_folder = os.path.join(exp_folder, 'pz004/20250918_1154_08/camera_20250918_1154_extracted')
    # dataset_root_folder = os.path.join(exp_folder, 'pz004/20250918_1156_18/camera_20250918_1156_extracted')
    # dataset_root_folder = os.path.join(exp_folder, 'pz004/20250918_1249_56/camera_20250918_1249_extracted')
    # dataset_root_folder = os.path.join(exp_folder, 'pz004/20250918_1252_07/camera_20250918_1252_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ kfar_galim 27.10.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20251027_kfar_galim'
    # dataset_root_folder = os.path.join(exp_folder, '20251027_123000/camera_20251027_1230_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ kfar_galim 26.11.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20251126_kfar_galim'
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1507_31/20251126_1510_20/camera_20251126_1610_extracted')  # no target
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1507_31/20251126_1511_16/camera_20251126_1611_extracted')  # no target
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1507_31/20251126_1513_05/camera_20251126_1613_extracted')  # no target
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1523_01/20251126_1526_12/camera_20251126_1626_extracted')  # passes far
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1536_04/20251126_1536_50/camera_20251126_1636_extracted')  # ends early
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1558_12/20251126_1600_23/camera_20251126_1700_extracted')  # no target
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1603_18/20251126_1604_48/camera_20251126_1704_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1603_18/20251126_1608_34/camera_20251126_1708_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ kfar_galim 04.12.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20251204_kfar_galim'
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1453_40/20251204_1454_07/camera_20251204_1554_extracted')  # no target
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1511_13/20251204_1511_34/camera_20251204_1611_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1511_13/20251204_1515_28/camera_20251204_1615_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1526_11/20251204_1527_23/camera_20251204_1627_extracted')  # passes far
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1526_11/20251204_1532_30/camera_20251204_1632_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1550_34/20251204_1550_54/camera_20251204_1650_extracted')  # passes far
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1600_55/20251204_1601_20/camera_20251204_1701_extracted')  # ends early
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1600_55/20251204_1603_11/camera_20251204_1703_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 08.12.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20251208_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1235_31/20251208_1239_35/camera_20251208_1339_extracted')  # no target
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1246_16/20251208_1247_59/camera_20251208_1348_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1256_17/20251208_1301_26/camera_20251208_1401_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1256_17/20251208_1303_25/camera_20251208_1403_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1327_30/20251208_1332_56/camera_20251208_1433_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1454_32/20251208_1458_20/camera_20251208_1558_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1454_32/20251208_1500_28/camera_20251208_1600_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1509_37/20251208_1513_17/camera_20251208_1613_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1509_37/20251208_1515_50/camera_20251208_1615_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1521_09/20251208_1523_58/camera_20251208_1624_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1521_09/20251208_1526_27/camera_20251208_1626_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 14.12.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20251214_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20251214_1322_23/20251214_1327_07/camera_20251214_1427_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251214_1232_41/20251214_1233_08/camera_20251214_1333_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20251214_1232_41/20251214_1235_55/camera_20251214_1336_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 18.02.2026 ------------------------------ (18" quadrotor target)
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20260218_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20260218_1019_25/20260218_1020_18/camera_20260218_1120_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260218_1123_37/20260218_1124_44/camera_20260218_1225_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260218_1306_42/20260218_1307_22/camera_20260218_1407_extracted')  # only far target
    # dataset_root_folder = os.path.join(exp_folder, '20260218_1306_42/20260218_1309_26/camera_20260218_1409_extracted')  # only far target
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 22.03.2026 ------------------------------ (10" quadrotor target)
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20260322_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20260322_1459_23/camera_20260322_1559_f1_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260322_1459_23/camera_20260322_1559_f2_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260322_1459_23/camera_20260322_1559_f3_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 24.03.2026 ------------------------------ (10" quadrotor target)
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20260324_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1029_26/20260324_1031_04/camera_20260324_1131_f1_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1029_26/20260324_1031_04/camera_20260324_1131_f2_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1029_26/20260324_1031_04/camera_20260324_1131_f3_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1029_26/20260324_1031_04/camera_20260324_1131_f4_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1037_17/20260324_1037_40/camera_20260324_1137_f1_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1037_17/20260324_1037_40/camera_20260324_1137_f2_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1037_17/20260324_1037_40/camera_20260324_1137_f3_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1037_17/20260324_1042_30/camera_20260324_1142_f1_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1037_17/20260324_1042_30/camera_20260324_1142_f2_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1037_17/20260324_1042_30/camera_20260324_1142_f3_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1048_29/20260324_1049_06/camera_20260324_1149_f1_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260324_1048_29/20260324_1049_06/camera_20260324_1149_f2_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 25.06.2026 ------------------------------ (10" quadrotor target)
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20260325_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307_f1_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307_f2_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307_f3_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307_f4_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307_f5_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 29.03.2026 ------------------------------ (10" quadrotor target)
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20260329_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20260329_1404_43/20260329_1407_02/camera_20260329_1407_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260329_1404_43/20260329_1405_07/camera_20260329_1405_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260329_1411_41/20260329_1411_56/camera_20260329_1412_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260329_1445_19/20260329_1446_24/camera_20260329_1446_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260329_1445_19/20260329_1448_12/camera_20260329_1448_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260329_1516_47/20260329_1518_30/camera_20260329_1518_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ mashabei_sade 31.03.2026 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20260331_mashabei_sade'
    # dataset_root_folder = os.path.join(exp_folder, '20260331_0747_53/20260331_0751_33/camera_20260331_0751_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260331_1400_31/20260331_1403_27/camera_20260331_1403_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260331_1400_31/20260331_1406_56/camera_20260331_1407_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260331_1419_28/20260331_1422_32/camera_20260331_1422_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% thermal dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # ------------------ reshafim 01.02.2026 thermal ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/thermal_experiments/20260201_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20260201_1408_32/camera_20260201_1508_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 02.02.2026 thermal ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/thermal_experiments/20260202_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20260202_1016_47/20260202_1017_32/camera_20260202_1117_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260202_1016_47/20260202_1020_41/camera_20260202_1120_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260202_1044_13/20260202_1046_28/camera_20260202_1146_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260202_1044_13/20260202_1048_30/camera_20260202_1148_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 05.02.2026 thermal ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/thermal_experiments/20260205_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20260205_1030_44/20260205_1032_58/camera_20260205_1133_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260205_1123_17/20260205_1125_01/camera_20260205_1225_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 06.02.2026 thermal ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/thermal_experiments/20260206_reshafim'
    # dataset_root_folder = os.path.join(exp_folder, '20260206_1552_14/camera_20260206_1652_extracted')
    # metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ reshafim 15.04.2026 thermal ------------------------------
    exp_folder = '/home/roee/Projects/datasets/interceptor_drone/thermal_experiments/20260415_reshafim'
    dataset_root_folder = os.path.join(exp_folder, '20260415_1202_12/20260415_1204_46/camera_20260415_1204_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1202_12/20260415_1206_29/camera_20260415_1206_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1202_12/20260415_1208_44/camera_20260415_1208_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1336_11/20260415_1338_45/camera_20260415_1338_f1_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1336_11/20260415_1338_45/camera_20260415_1338_f2_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1336_11/20260415_1338_45/camera_20260415_1338_f3_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1336_11/20260415_1341_48/camera_20260415_1341_f1_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1336_11/20260415_1341_48/camera_20260415_1341_f2_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1403_20/20260415_1405_11/camera_20260415_1405_f1_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1403_20/20260415_1405_11/camera_20260415_1405_f2_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1403_20/20260415_1405_11/camera_20260415_1405_f3_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1403_20/20260415_1407_24/camera_20260415_1407_f1_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1403_20/20260415_1407_24/camera_20260415_1407_f2_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1403_20/20260415_1407_24/camera_20260415_1407_f3_extracted')
    # dataset_root_folder = os.path.join(exp_folder, '20260415_1403_20/20260415_1407_24/camera_20260415_1407_f4_extracted')
    metadata_yaml_file = os.path.join(dataset_root_folder, 'dataset_metadata.yaml')

    # ------------------ 20260419_reshafim thermal ------------------------------
    exp_folder = '/home/roee/Projects/datasets/interceptor_drone/thermal_experiments/20260419_reshafim'
    # base_folder = os.path.join(exp_folder, '20260419_0918_56/20260419_0920_06/camera_20260419_0920_extracted')  # bad magic
    # base_folder = os.path.join(exp_folder, '20260419_0918_56/20260419_0921_15/camera_20260419_0921_extracted')  # bad magic
    # base_folder = os.path.join(exp_folder, '20260419_0918_56/20260419_0928_10/camera_20260419_0928_extracted')  # what to do with the bird?
    # base_folder = os.path.join(exp_folder, '20260419_0936_00/20260419_0937_08/camera_20260419_0937_extracted')  # bad magic
    base_folder = os.path.join(exp_folder, '20260419_0936_00/20260419_0939_18/camera_20260419_0939_extracted')
    color_space = 'Y16'


    # ----------------------- add dataset metadata ---------------------
    json_path = str(Path(dataset_root_folder) / 'annotations' / 'coco_dataset.json')
    dataset = coco_dataset_manager.CocoDatasetManager()
    dataset.load_coco(json_path)

    fix_annotations_bbox(dataset)

    ret, metadata = load_scenario_metadata(metadata_yaml_file)

    if ret and 'images' in metadata.keys():
        images_metadata = metadata['images']
        add_images_common_metadata(dataset, images_metadata)
    else:
        raise Exception('No images found in metadata')

    if ret and 'annotations' in metadata.keys():
        annotations_metadata = metadata['annotations']
        add_annotations_common_metadata(dataset, None, annotations_metadata)
    else:
        raise Exception('No annotations found in metadata')

    # save dataset
    output_json_file_name = 'coco_dataset.json'
    dataset.save_coco(dataset.root_folder, output_json_file_name, copy_images=False, overwrite=False)

    print('Done!')