import os
from pathlib import Path
import cv2
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

if __name__ == '__main__':


    # ------------------ 20251214_reshafim ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251214_reshafim'
    # images_metadata = {'datetime': None, 'location': 'reshafim',
    #                    'camera': 'arducam_imx708_B0310',
    #                    'clouds': 'clear', 'visibility': 'haze', 'daytime': 'day', 'lighting': 'good',
    #                    'view_direction':'north', 'sun_direction': 'back'}
    # annotations_metadata = {'subclass': 'fms_1800mm_ranger_green'}
    #
    # dataset_root_folder = os.path.join(exp_folder, '20251214_1322_23/20251214_1327_07/camera_20251214_1427_extracted')
    # images_metadata['datetime'] = '20251214_132707'
    # dataset_root_folder = os.path.join(exp_folder, '20251214_1232_41/20251214_1233_08/camera_20251214_1333_extracted')
    # images_metadata['datetime'] = '20251214_123308'
    # dataset_root_folder = os.path.join(exp_folder, '20251214_1232_41/20251214_1235_55/camera_20251214_1336_extracted')
    # images_metadata['datetime'] = '20251214_123555'


    # ------------------ 20251208_reshafim ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251208_reshafim'
    # images_metadata = {'datetime': None, 'location': 'reshafim',
    #                    'camera': 'arducam_imx708_B0310',
    #                    'clouds': 'mostly_cloudy', 'visibility': 'clear', 'daytime': 'day', 'lighting': 'good',
    #                    'view_direction':'north', 'sun_direction': 'back'}
    # annotations_metadata = {'subcategory': 'fms_1800mm_ranger_green'}

    # dataset_root_folder = os.path.join(exp_folder, '20251208_1235_31/20251208_1239_35/camera_20251208_1339_extracted')  # no target
    # images_metadata['datetime'] = '20251208_123935'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1246_16/20251208_1247_59/camera_20251208_1348_extracted')
    # images_metadata['datetime'] = '20251208_124759'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1256_17/20251208_1301_26/camera_20251208_1401_extracted')
    # images_metadata['datetime'] = '20251208_130126'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1256_17/20251208_1303_25/camera_20251208_1403_extracted')
    # images_metadata['datetime'] = '20251208_130325'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1327_30/20251208_1332_56/camera_20251208_1433_extracted')
    # images_metadata['datetime'] = '20251208_133256'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1454_32/20251208_1458_20/camera_20251208_1558_extracted')
    # images_metadata['datetime'] = '20251208_145820'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1454_32/20251208_1500_28/camera_20251208_1600_extracted')
    # images_metadata['datetime'] = '20251208_150028'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1509_37/20251208_1513_17/camera_20251208_1613_extracted')
    # images_metadata['datetime'] = '20251208_151317'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1509_37/20251208_1515_50/camera_20251208_1615_extracted')
    # images_metadata['datetime'] = '20251208_151550'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1521_09/20251208_1523_58/camera_20251208_1624_extracted')
    # images_metadata['datetime'] = '20251208_152358'
    # dataset_root_folder = os.path.join(exp_folder, '20251208_1521_09/20251208_1526_27/camera_20251208_1626_extracted')
    # images_metadata['datetime'] = '20251208_152627'


    # ------------------ kfar_galim 04.12.2025 ------------------------------
    exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251204_kfar_galim'
    images_metadata = {'datetime': None, 'location': 'kfar_galim',
                       'camera': 'arducam_imx708_B0310',
                       'clouds': 'mostly_clear', 'visibility': 'clear', 'daytime': 'day', 'lighting': 'good',
                       'view_direction':'north_west', 'sun_direction': 'left'}
    annotations_metadata = {'subcategory': 'fms_1800mm_ranger_green'}

    # dataset_root_folder = os.path.join(exp_folder, '20251204_1453_40/20251204_1454_07/camera_20251204_1554_extracted')  # no target
    # images_metadata['datetime'] = '20251204_145407'
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1511_13/20251204_1511_34/camera_20251204_1611_extracted')
    # images_metadata['datetime'] = '20251204_151134'
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1511_13/20251204_1515_28/camera_20251204_1615_extracted')
    # images_metadata['datetime'] = '20251204_151528'
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1526_11/20251204_1527_23/camera_20251204_1627_extracted')  # passes far
    # images_metadata['datetime'] = '20251204_152723'
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1526_11/20251204_1532_30/camera_20251204_1632_extracted')
    # images_metadata['datetime'] = '20251204_153230'
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1550_34/20251204_1550_54/camera_20251204_1650_extracted')  # passes far
    # images_metadata['datetime'] = '20251204_155054'
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1600_55/20251204_1601_20/camera_20251204_1701_extracted')  # ends early
    # images_metadata['datetime'] = '20251204_160120'
    # dataset_root_folder = os.path.join(exp_folder, '20251204_1600_55/20251204_1603_11/camera_20251204_1703_extracted')
    # images_metadata['datetime'] = '20251204_160311'


    # ------------------ kfar_galim 26.11.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251126_kfar_galim'
    # images_metadata = {'datetime': None, 'location': 'kfar_galim',
    #                    'camera': 'arducam_imx708_B0310',
    #                    'clouds': 'clear', 'visibility': 'clear', 'daytime': 'evening', 'lighting': 'low',
    #                    'view_direction':'north_west', 'sun_direction': 'left_front'}
    # annotations_metadata = {'subcategory': 'fms_1800mm_ranger_green'}

    # dataset_root_folder = os.path.join(exp_folder, '20251126_1507_31/20251126_1510_20/camera_20251126_1610_extracted')  # no target
    # images_metadata['datetime'] = '20251126_151020'
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1507_31/20251126_1511_16/camera_20251126_1611_extracted')  # no target
    # images_metadata['datetime'] = '20251126_151116'
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1507_31/20251126_1513_05/camera_20251126_1613_extracted')  # no target
    # images_metadata['datetime'] = '20251126_151305'
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1523_01/20251126_1526_12/camera_20251126_1626_extracted')  # passes far
    # images_metadata['datetime'] = '20251126_152612'
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1536_04/20251126_1536_50/camera_20251126_1636_extracted')  # ends early
    # images_metadata['datetime'] = '20251126_153650'
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1558_12/20251126_1600_23/camera_20251126_1700_extracted')  # no target
    # images_metadata['datetime'] = '20251126_160023'
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1603_18/20251126_1604_48/camera_20251126_1704_extracted')
    # images_metadata['datetime'] = '20251126_160448'
    # dataset_root_folder = os.path.join(exp_folder, '20251126_1603_18/20251126_1608_34/camera_20251126_1708_extracted')
    # images_metadata['datetime'] = '20251126_160834'


    # ------------------ kfar_galim 27.10.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251027_kfar_galim'
    # images_metadata = {'datetime': None, 'location': 'kfar_galim',
    #                    'camera': 'arducam_imx708_B0310',
    #                    'clouds': 'clear', 'visibility': 'clear', 'daytime': 'day', 'lighting': 'good',
    #                    'view_direction':'north_west', 'sun_direction': 'back'}
    # annotations_metadata = {'subcategory': 'E-Flite_Apprentice_Sts1.5M'}
    # dataset_root_folder = os.path.join(exp_folder, '20251027_123000/camera_20251027_1230_extracted')
    # images_metadata['datetime'] = '20251027_123000'


    # ------------------ lehavim 17.09.2025 - pre test ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250917_lehavim'
    # images_metadata = {'datetime': None, 'location': 'lehavim',
    #                    'camera': 'arducam_imx708_B0310',
    #                    'clouds': 'mostly_clear', 'visibility': 'clear', 'daytime': 'day', 'lighting': 'good',
    #                    'view_direction': 'west', 'sun_direction': 'left'}
    # annotations_metadata = {'subcategory': 'E-Flite_Apprentice_Sts1.5M'}
    #
    # dataset_root_folder = os.path.join(exp_folder, 'hb003/20250915_1028_24/camera_20250915_1028_extracted')  # no target
    # images_metadata['datetime'] = '20250915_102824'


    # ------------------ lehavim 17.09.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250917_lehavim'
    # images_metadata = {'datetime': None, 'location': 'lehavim',
    #                    'camera': 'arducam_imx708_B0310',
    #                    'clouds': 'mostly_clear', 'visibility': 'clear', 'daytime': 'afternoon', 'lighting': 'good',
    #                    'view_direction': 'west', 'sun_direction': 'front'}
    # annotations_metadata = {'subcategory': 'E-Flite_Apprentice_Sts1.5M'}
    #
    # dataset_root_folder = os.path.join(exp_folder, 'hb003/20250917_1747_34/camera_20250917_1747_extracted')
    # images_metadata['datetime'] = '20250917_174734'


    # ------------------ lehavim 18.09.2025 pre test  ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250918_lehavim'
    # images_metadata['datetime'] = '20251027_123000'
    # images_metadata = {'datetime': None, 'location': 'kfar_galim',
    #                    'camera': 'arducam_imx708_B0310',
    #                    'clouds': 'mostly_clear', 'visibility': 'clear', 'daytime': 'day', 'lighting': 'good',
    #                    'view_direction': 'west', 'sun_direction': 'left'}
    # annotations_metadata = {'subcategory': 'E-Flite_Apprentice_Sts1.5M'}
    # dataset_root_folder = os.path.join(exp_folder, 'hb002/20250915_1019_36/camera_20250915_1019_extracted')  # no target
    # images_metadata['datetime'] = '20250915_101936'


    # ------------------ lehavim 18.09.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250918_lehavim'
    # images_metadata = {'datetime': None, 'location': 'lehavim',
    #                    'camera': 'arducam_imx708_B0310',
    #                    'clouds': 'mostly_clear', 'visibility': 'haze', 'daytime': 'day', 'lighting': 'good',
    #                    'view_direction': 'north_west', 'sun_direction': 'front_left'}
    # annotations_metadata = {'subcategory': 'E-Flite_Apprentice_Sts1.5M'}

    # dataset_root_folder = os.path.join(exp_folder, 'hb002/20250918_1040_02/camera_20250918_1040_extracted')  # no target
    # images_metadata['datetime'] = '20250918_104002'
    # dataset_root_folder = os.path.join(exp_folder, 'hb002/20250918_1041_26/camera_20250918_1041_extracted')  # no target
    # images_metadata['datetime'] = '20250918_104126'
    # dataset_root_folder = os.path.join(exp_folder, 'hb002/20250918_1044_49/camera_20250918_1044_extracted')
    # images_metadata['datetime'] = '20250918_104449'
    # dataset_root_folder = os.path.join(exp_folder, 'hb004/20250918_1451_40/camera_20250918_1451_extracted')
    # images_metadata['datetime'] = '20250918_145140'
    # dataset_root_folder = os.path.join(exp_folder, 'hb004/20250918_1453_47/camera_20250918_1453_extracted')
    # images_metadata['datetime'] = '20250918_145347'
    # dataset_root_folder = os.path.join(exp_folder, 'hb004/20250918_1456_00/camera_20250918_1456_extracted')
    # images_metadata['datetime'] = '20250918_145600'
    # dataset_root_folder = os.path.join(exp_folder, 'pz004/20250918_1154_08/camera_20250918_1154_extracted')
    # images_metadata['datetime'] = '20250918_115408'
    # dataset_root_folder = os.path.join(exp_folder, 'pz004/20250918_1156_18/camera_20250918_1156_extracted')
    # images_metadata['datetime'] = '20250918_115618'
    # dataset_root_folder = os.path.join(exp_folder, 'pz004/20250918_1249_56/camera_20250918_1249_extracted')
    # images_metadata['datetime'] = '20250918_124956'
    # dataset_root_folder = os.path.join(exp_folder, 'pz004/20250918_1252_07/camera_20250918_1252_extracted')
    # images_metadata['datetime'] = '20250918_125207'


    # ------------------ kfar_masarik 08.06.2025 ------------------------------
    # exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250608_kfar_masarik'
    # images_metadata = {'datetime': None, 'location': 'kfar_masarik',
    #                    'camera': 'arducam_imx708_B0310',
    #                    'clouds': 'mostly_clear', 'visibility': 'haze', 'daytime': 'sunset', 'lighting': 'low',
    #                    'view_direction': 'north', 'sun_direction': 'left'}
    # annotations_metadata = {'subcategory': 'E-Flite_Apprentice_Sts1.5M'}

    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-04-53/camera_2025_6_8-15_4_56_extracted')
    # images_metadata['datetime'] = '20250608_180453'
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-29-51/camera_2025_6_8-15_29_54_extracted')  # passes farK
    # images_metadata['datetime'] = '20250608_182951'
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-51-15/camera_2025_6_8-15_51_18_extracted')
    # images_metadata['datetime'] = '20250608_185115'
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-52-56/camera_2025_6_8-15_52_58_extracted')  # passes far
    # images_metadata['datetime'] = '20250608_185256'
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-53-46/camera_2025_6_8-15_53_49_extracted')  # passes far
    # images_metadata['datetime'] = '20250608_185346'
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_18-59-46/camera_2025_6_8-15_59_49_extracted')  # passes far
    # images_metadata['datetime'] = '20250608_185946'
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_19-00-25/camera_2025_6_8-16_0_28_extracted')  # passes far
    # images_metadata['datetime'] = '20250608_190025'
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_19-08-34/camera_2025_6_8-16_8_48_extracted')
    # images_metadata['datetime'] = '20250608_190834'
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_19-09-33/camera_2025_6_8-16_9_38_extracted')
    # images_metadata['datetime'] = '20250608_190933'
    # dataset_root_folder = os.path.join(exp_folder, '2025-06-08_19-25-35/camera_2025_6_8-16_25_38_extracted')  # passes far
    # images_metadata['datetime'] = '20250608_192535'


    # ------------------ hadera 21.04.2025 ------------------------------
    exp_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250421_hadera'
    images_metadata = {'datetime': None, 'location': 'hadera',
                       'camera': 'arducam_imx708_B0310',
                       'clouds': 'clear', 'visibility': 'good', 'daytime': 'day', 'lighting': 'good',
                       'view_direction': 'south', 'sun_direction': 'front'}
    annotations_metadata = {'subcategory': 'E-Flite_Apprentice_Sts1.5M_blackened'}

    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_10-57-38/camera_2025_4_21-7_59_8_extracted')  # no target (almost)
    # images_metadata['datetime'] = '20250421_105738'
    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_10-59-30/camera_2025_4_21-7_59_41_extracted')  # OK
    # images_metadata['datetime'] = '20250421_105930'
    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_11-11-25/camera_2025_4_21-8_12_28_extracted')  # OK
    # images_metadata['datetime'] = '20250421_111125'
    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_11-13-03/camera_2025_4_21-8_13_40_extracted')  # OK
    # images_metadata['datetime'] = '20250421_111303'
    # dataset_root_folder = os.path.join(exp_folder, '2025-04-21_11-24-25/camera_2025_4_21-8_25_32_extracted')  # passes far - OK
    # images_metadata['datetime'] = '20250421_112425'
    dataset_root_folder = os.path.join(exp_folder, '2025-04-21_11-26-29/camera_2025_4_21-8_26_42_extracted')  # passes far - OK
    images_metadata['datetime'] = '20250421_112629'



    # ----------------------- add dataset metadata ---------------------
    json_path = str(Path(dataset_root_folder) / 'annotations' / 'coco_dataset.json')
    dataset = coco_dataset_manager.CocoDatasetManager()
    dataset.load_coco(json_path)

    fix_annotations_bbox(dataset)
    add_images_common_metadata(dataset, images_metadata)
    add_annotations_common_metadata(dataset, None, annotations_metadata)

    # save dataset
    output_json_file_name = 'coco_dataset.json'
    dataset.save_coco(dataset.root_folder, output_json_file_name, copy_images=False, overwrite=False)

    print('Done!')