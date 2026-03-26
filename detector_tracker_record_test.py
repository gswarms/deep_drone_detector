""" Detect and track blobs in image
"""
import os
import cv2
import sys
import numpy as np
from test_utils.standard_record import StandardRecord
from test_utils.roi_utils import PolygonPerFrame
from test_utils import common_utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import detector_tracker

if __name__ == '__main__':

    # ------------------ kfar massarik 08.06.2025 ------------------------------
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_17-30-42/camera_2025_6_8-14_30_56_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-04-53/camera_2025_6_8-15_4_56_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-17-57/camera_2025_6_8-15_18_8_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-29-51/camera_2025_6_8-15_29_54_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-51-15/camera_2025_6_8-15_51_18_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-52-56/camera_2025_6_8-15_52_58_extracted'  # ???
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-53-46/camera_2025_6_8-15_53_49_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-58-42/camera_2025_6_8-15_58_44_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-59-46/camera_2025_6_8-15_59_49_extracted'  # ???
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-00-25/camera_2025_6_8-16_0_28_extracted'  # ???
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-08-34/camera_2025_6_8-16_8_48_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-09-33/camera_2025_6_8-16_9_38_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-17-25/camera_2025_6_8-16_17_28_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-18-31/camera_2025_6_8-16_18_34_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-24-31/camera_2025_6_8-16_24_34_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-25-35/camera_2025_6_8-16_25_38_extracted'  # ???


    # ------------------ kfar galim 01.07.2025 ------------------------------
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_08-26-13/camera_2025_7_1-5_26_17_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-01-54/camera_2025_7_1-6_2_3_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-27-36/camera_2025_7_1-6_27_39_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-34-06/camera_2025_7_1-6_34_9_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-35-10/camera_2025_7_1-6_35_13_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-46-51/camera_2025_7_1-6_46_54_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-48-20/camera_2025_7_1-???_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-03-29/camera_2025_7_1-7_3_32_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-04-48/camera_2025_7_1-???_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-47-42/camera_2025_7_1-???_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-49-24/camera_2025_7_1-7_49_27_extracted'

    # ------------------ kfar galim 10.07.2025 ------------------------------
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_07-40-11/camera_2025_7_10-4_40_14_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_07-54-07/camera_2025_7_10-4_54_11_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_08-01-21/camera_2025_7_10-5_1_32_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_08-12-42/camera_2025_7_10-5_12_45_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_08-24-03/camera_2025_7_10-5_24_17_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_08-32-49/camera_2025_7_10-5_32_52_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_09-17-03/camera_2025_7_10-6_17_15_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_09-17-03/camera_2025_7_10-6_17_15_extracted'  # bad
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_09-17-49/camera_2025_7_10-6_17_53_extracted'  # bad


    # ------------------ kfar galim 16.07.2025 ------------------------------
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_09-18-14/camera_2025_7_16-6_18_18_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_09-18-58/camera_2025_7_16-6_19_1_extracted'


    # ------------------ kfar galim 30.07.2025 ------------------------------
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_07-59-08/camera_2025_7_30-4_59_23_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-13-53/camera_2025_7_30-5_13_58_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-24-20/camera_2025_7_30-5_24_24_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-25-35/camera_2025_7_30-5_25_39_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-42-38/camera_2025_7_30-5_42_41_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-48-07/camera_2025_7_30-5_48_17_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-50-03/camera_2025_7_30-5_50_7_extracted'
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_09-45-04/camera_2025_7_30-6_45_8_extracted'  # bad bag


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5 common dataset start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    # ------------------ 20251214_reshafim ------------------------------  *** clr_format = 'BGR'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251214_reshafim'
    # record_folder = os.path.join(base_folder, '20251214_1124_19/20251214_1234_56/camera_20251214_1335_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251214_1232_41/20251214_1233_08/camera_20251214_1333_extracted')
    # record_folder = os.path.join(base_folder, '20251214_1232_41/20251214_1234_56/camera_20251214_1335_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251214_1232_41/20251214_1235_55/camera_20251214_1336_extracted')
    # record_folder = os.path.join(base_folder, '20251214_1232_41/20251214_1236_52/camera_20251214_1336_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251214_1322_23/20251214_1325_35/camera_20251214_1425_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251214_1322_23/20251214_1327_07/camera_20251214_1427_extracted')


    # ------------------ 20251208_reshafim ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251208_reshafim'
    # record_folder = os.path.join(base_folder, '20251208_1155_58/20251208_1157_42/camera_20251208_1257_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1155_58/20251208_1200_23/camera_20251208_1300_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1155_58/20251208_1204_12/camera_20251208_1304_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1155_58/20251208_1206_00/camera_20251208_1306_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1155_58/20251208_1207_52/camera_20251208_1307_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1216_03/20251208_1216_29/camera_20251208_1316_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1235_31/20251208_1239_35/camera_20251208_1339_extracted')
    # record_folder = os.path.join(base_folder, '20251208_1246_16/20251208_1247_59/camera_20251208_1348_extracted')
    # record_folder = os.path.join(base_folder, '20251208_1256_17/20251208_1257_05/camera_20251208_1357_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1256_17/20251208_1259_11/camera_20251208_1359_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1256_17/20251208_1301_26/camera_20251208_1401_extracted')
    # record_folder = os.path.join(base_folder, '20251208_1256_17/20251208_1303_25/camera_20251208_1403_extracted')
    # record_folder = os.path.join(base_folder, '20251208_1327_30/20251208_1332_56/camera_20251208_1433_extracted')
    # record_folder = os.path.join(base_folder, '20251208_1454_32/20251208_1457_05/camera_20251208_1557_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1454_32/20251208_1458_20/camera_20251208_1558_extracted')
    # record_folder = os.path.join(base_folder, '20251208_1454_32/20251208_1500_28/camera_20251208_1600_extracted')
    # record_folder = os.path.join(base_folder, '20251208_1454_32/20251208_1501_47/camera_20251208_1601_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1509_37/20251208_1511_41/camera_20251208_1611_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1509_37/20251208_1513_17/camera_20251208_1613_extracted')
    # record_folder = os.path.join(base_folder, '20251208_1509_37/20251208_1514_28/camera_20251208_1614_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1509_37/20251208_1515_50/camera_20251208_1615_extracted')
    # record_folder = os.path.join(base_folder, '20251208_1509_37/20251208_1516_53/camera_20251208_1616_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1521_09/20251208_1523_58/camera_20251208_1624_extracted')
    # record_folder = os.path.join(base_folder, '20251208_1521_09/20251208_1525_11/camera_20251208_1625_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251208_1521_09/20251208_1526_27/camera_20251208_1626_extracted')  # test mns ????????


    # ------------------ kfar_galim 04.12.2025 ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251204_kfar_galim'
    # record_folder = os.path.join(base_folder, '20251204_1423_16/20251204_1423_36/camera_20251204_1523_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251204_1423_16/20251204_1425_28/camera_20251204_1525_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251204_1423_16/20251204_1431_50/camera_20251204_1531_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251204_1438_06/20251204_1438_28/camera_20251204_1538_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251204_1453_40/20251204_1454_07/camera_20251204_1554_extracted')
    # record_folder = os.path.join(base_folder, '20251204_1511_13/20251204_1511_34/camera_20251204_1611_extracted')
    # record_folder = os.path.join(base_folder, '20251204_1511_13/20251204_1515_28/camera_20251204_1615_extracted')
    # record_folder = os.path.join(base_folder, '20251204_1526_11/20251204_1527_23/camera_20251204_1627_extracted')
    # record_folder = os.path.join(base_folder, '20251204_1526_11/20251204_1532_30/camera_20251204_1632_extracted')
    # record_folder = os.path.join(base_folder, '20251204_1550_34/20251204_1550_54/camera_20251204_1650_extracted')
    # record_folder = os.path.join(base_folder, '20251204_1600_55/20251204_1601_20/camera_20251204_1701_extracted')
    # record_folder = os.path.join(base_folder, '20251204_1600_55/20251204_1603_11/camera_20251204_1703_extracted')


    # ------------------ kfar_galim 26.11.2025 ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251126_kfar_galim'
    # record_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1509_36/camera_20251126_1609_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1510_20/camera_20251126_1610_extracted')
    # record_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1511_16/camera_20251126_1611_extracted')
    # record_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1512_12/camera_20251126_1612_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1513_05/camera_20251126_1613_extracted')
    # record_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1514_12/camera_20251126_1614_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251126_1523_01/20251126_1524_04/camera_20251126_1624_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251126_1523_01/20251126_1526_12/camera_20251126_1626_extracted')
    # record_folder = os.path.join(base_folder, '20251126_1523_01/20251126_1527_11/camera_20251126_1627_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251126_1536_04/20251126_1536_50/camera_20251126_1636_extracted')
    # record_folder = os.path.join(base_folder, '20251126_1558_12/20251126_1558_24/camera_20251126_1658_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251126_1558_12/20251126_1600_23/camera_20251126_1700_extracted')
    # record_folder = os.path.join(base_folder, '20251126_1603_18/20251126_1603_41/camera_20251126_1703_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251126_1603_18/20251126_1604_48/camera_20251126_1704_extracted')
    # record_folder = os.path.join(base_folder, '20251126_1603_18/20251126_1605_59/camera_20251126_1706_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251126_1603_18/20251126_1607_28/camera_20251126_1707_extracted')  # bad
    # record_folder = os.path.join(base_folder, '20251126_1603_18/20251126_1608_34/camera_20251126_1708_extracted')

    # ------------------ kfar_galim 27.10.2025 ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251027_kfar_galim'
    # record_folder = os.path.join(base_folder, '20251027_123000/camera_20251027_1230_extracted')


    # ------------------ lehavim 18.09.2025 ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250918_lehavim'

    # record_folder = os.path.join(base_folder, 'hb002/20250910_1441_06/camera_20250910_1441_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb002/20250910_1442_03/camera_20250910_1442_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb002/20250915_1019_36/camera_20250915_1019_extracted')
    # record_folder = os.path.join(base_folder, 'hb002/20250918_0941_14/camera_20250918_0941_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb002/20250918_1005_37/camera_20250918_1005_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb002/20250918_1008_19/camera_20250918_1008_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb002/20250918_1010_22/camera_20250918_1010_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb002/20250918_1040_02/camera_20250918_1040_extracted')
    # record_folder = os.path.join(base_folder, 'hb002/20250918_1041_26/camera_20250918_1041_extracted')
    # record_folder = os.path.join(base_folder, 'hb002/20250918_1043_09/camera_20250918_1043_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb002/20250918_1044_49/camera_20250918_1044_extracted')
    # record_folder = os.path.join(base_folder, 'hb002/20250918_1046_44/camera_20250918_1046_extracted')  # bad

    # record_folder = os.path.join(base_folder, 'hb004/20250918_1439_53/camera_20250918_1440_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb004/20250918_1441_41/camera_20250918_1441_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb004/20250918_1443_47/camera_20250918_1443_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb004/20250918_1446_11/camera_20250918_1446_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb004/20250918_1451_40/camera_20250918_1451_extracted')
    # record_folder = os.path.join(base_folder, 'hb004/20250918_1453_47/camera_20250918_1453_extracted')
    # record_folder = os.path.join(base_folder, 'hb004/20250918_1456_00/camera_20250918_1456_extracted')
    # record_folder = os.path.join(base_folder, 'hb004/20250918_1457_57/camera_20250918_1458_extracted')  # bad

    # record_folder = os.path.join(base_folder, 'pz004/20250918_1154_08/camera_20250918_1154_extracted')
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1156_18/camera_20250918_1156_extracted')
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1158_27/camera_20250918_1158_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1200_17/camera_20250918_1200_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1210_46/camera_20250918_1210_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1216_55/camera_20250918_1217_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1219_00/camera_20250918_1219_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1232_08/camera_20250918_1232_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1235_26/camera_20250918_1235_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1239_54/camera_20250918_1239_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1248_01/camera_20250918_1248_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1249_56/camera_20250918_1249_extracted')
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1252_07/camera_20250918_1252_extracted')
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1256_02/camera_20250918_1256_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1336_04/camera_20250918_1336_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1338_16/camera_20250918_1338_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1341_50/camera_20250918_1341_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'pz004/20250918_1410_23/camera_20250918_1411_extracted')  # bad

    # ------------------ lehavim 17.09.2025 ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250917_lehavim'
    # record_folder = os.path.join(base_folder, 'hb003/20250915_1028_24/camera_20250915_1028_extracted')
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1506_17/camera_20250917_1506_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1547_12/camera_20250917_1547_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1549_23/camera_20250917_1549_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1557_10/camera_20250917_1557_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1623_52/camera_20250917_1624_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1625_13/camera_20250917_1625_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1645_47/camera_20250917_1645_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1657_35/camera_20250917_1657_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1715_18/camera_20250917_1715_extracted')  # bad
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1747_34/camera_20250917_1747_extracted')
    # record_folder = os.path.join(base_folder, 'hb003/20250917_1754_26/camera_20250917_1754_extracted')  # bad

    # ------------------ kfar_masarik 08.06.2025 ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250608_kfar_masarik'
    # record_folder = os.path.join(base_folder, '2025-06-08_17-30-42/camera_2025_6_8-14_30_56_extracted')  # bad
    # record_folder = os.path.join(base_folder, '2025-06-08_18-04-53/camera_2025_6_8-15_4_56_extracted')
    # record_folder = os.path.join(base_folder, '2025-06-08_18-17-57/camera_2025_6_8-14_30_56_extracted')  # bad
    # record_folder = os.path.join(base_folder, '2025-06-08_18-17-57/camera_2025_6_8-15_18_8_extracted')  # bad
    # record_folder = os.path.join(base_folder, '2025-06-08_18-29-51/camera_2025_6_8-15_29_54_extracted')
    # record_folder = os.path.join(base_folder, '2025-06-08_18-41-19/camera_2025_6_8-15_41_22_extracted')  # bad
    # record_folder = os.path.join(base_folder, '2025-06-08_18-45-39/camera_2025_6_8-15_45_44_extracted')  # bad
    # record_folder = os.path.join(base_folder, '2025-06-08_18-51-15/camera_2025_6_8-15_51_18_extracted')
    # record_folder = os.path.join(base_folder, '2025-06-08_18-52-56/camera_2025_6_8-15_52_58_extracted')
    # record_folder = os.path.join(base_folder, '2025-06-08_18-53-46/camera_2025_6_8-15_53_49_extracted')
    # record_folder = os.path.join(base_folder, '2025-06-08_18-58-42/camera_2025_6_8-15_58_44_extracted')  # bad
    # record_folder = os.path.join(base_folder, '2025-06-08_18-59-46/camera_2025_6_8-15_59_49_extracted')
    # record_folder = os.path.join(base_folder, '2025-06-08_19-00-25/camera_2025_6_8-16_0_28_extracted')
    # record_folder = os.path.join(base_folder, '2025-06-08_19-08-34/camera_2025_6_8-16_8_48_extracted')
    # record_folder = os.path.join(base_folder, '2025-06-08_19-09-33/camera_2025_6_8-16_9_38_extracted')
    # record_folder = os.path.join(base_folder, '2025-06-08_19-17-25/camera_2025_6_8-16_17_28_extracted')  # bad
    # record_folder = os.path.join(base_folder, '2025-06-08_19-18-31/camera_2025_6_8-16_18_34_extracted')  # bad
    # record_folder = os.path.join(base_folder, '2025-06-08_19-24-31/camera_2025_6_8-16_24_34_extracted')  # bad
    # record_folder = os.path.join(base_folder, '2025-06-08_19-25-35/camera_2025_6_8-16_25_38_extracted')

    # ------------------ hadera 21.04.2025 ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250421_hadera'
    # record_folder = os.path.join(base_folder, '2025-04-21_10-57-38/camera_2025_4_21-7_59_8_extracted')
    # record_folder = os.path.join(base_folder, '2025-04-21_10-59-30/camera_2025_4_21-7_59_41_extracted')
    # record_folder = os.path.join(base_folder, '2025-04-21_11-11-25/camera_2025_4_21-8_12_28_extracted')
    # record_folder = os.path.join(base_folder, '2025-04-21_11-13-03/camera_2025_4_21-8_13_40_extracted')
    # record_folder = os.path.join(base_folder, '2025-04-21_11-24-25/camera_2025_4_21-8_25_32_extracted')
    # record_folder = os.path.join(base_folder, '2025-04-21_11-26-29/camera_2025_4_21-8_26_42_extracted')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5 common dataset end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    # ------------------ kfar galim 27.10.2025 ------------------------------
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20251027_kfar_galim/20251027_123000/camera_20251027_1230_extracted'


    # ------------------ reshafim 20.01.2026 ------------------------------
    # record_folder = '/home/roee/Projects/datasets/interceptor_drone/20260120_reshafim/20260120_1029_46/20260120_103322/camera_20260120_1133_extracted'


    # ------------------ 309 25.02.2026 ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260225_309'
    # record_folder = os.path.join(base_folder, '20260225_0911_49/20260225_091227/camera_20260225_1012_extracted')
    # record_folder = os.path.join(base_folder, '20260225_0911_49/20260225_091409/camera_20260225_1014_extracted')  # bad bag magic
    # record_folder = os.path.join(base_folder, '20260225_0911_49/20260225_091522/camera_20260225_1015_extracted')
    # record_folder = os.path.join(base_folder, '20260225_0911_49/20260225_091409/camera_20260225_1014_extracted')  # bad bag magic
    # record_folder = os.path.join(base_folder, '20260225_0911_49/20260225_091522/camera_20260225_1015_extracted')
    # record_folder = os.path.join(base_folder, '20260225_1233_44/20260225_123437/camera_20260225_1334_extracted')  # bad bag magic
    # record_folder = os.path.join(base_folder, '20260225_1233_44/20260225_123557/camera_20260225_1336_extracted')  # bad bag magic
    # record_folder = os.path.join(base_folder, '20260225_1233_44/20260225_123723/camera_20260225_1337_extracted')
    # record_folder = os.path.join(base_folder, '20260225_1347_29/20260225_134854/camera_20260225_1448_extracted')
    # record_folder = os.path.join(base_folder, '20260225_1347_29/20260225_135130/camera_20260225_1451_extracted')  # bad bag magic



    # ------------------ 20260322_reshafim ------------------------------ (10" quadrotor target)
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20260322_reshafim'
    # record_folder = os.path.join(base_folder, '20260322_1459_23/camera_20260322_1559_f1_extracted')
    # record_folder = os.path.join(base_folder, '20260322_1459_23/camera_20260322_1559_f2_extracted')
    # record_folder = os.path.join(base_folder, '20260322_1459_23/camera_20260322_1559_f3_extracted')

    # ------------------ 20260218_reshafim ------------------------------ (18" quadrotor target)
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20260218_reshafim'
    # record_folder = os.path.join(base_folder, '20260218_1019_25/20260218_1020_18/camera_20260218_1120_extracted')
    # record_folder = os.path.join(base_folder, '20260218_1123_37/20260218_1124_44/camera_20260218_1225_extracted')
    # record_folder = os.path.join(base_folder, '20260218_1306_42/20260218_1307_22/camera_20260218_1407_extracted')  # only far target
    # record_folder = os.path.join(base_folder, '20260218_1306_42/20260218_1309_26/camera_20260218_1409_extracted')  # only far target

    # ------------------ 20260324_reshafim ------------------------------ (10" quadrotor target)
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20260324_reshafim'
    # record_folder = os.path.join(base_folder, '20260324_1029_26/20260324_1031_04/camera_20260324_1131_f1_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1029_26/20260324_1031_04/camera_20260324_1131_f2_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1029_26/20260324_1031_04/camera_20260324_1131_f3_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1029_26/20260324_1031_04/camera_20260324_1131_f4_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1037_17/20260324_1037_40/camera_20260324_1137_f1_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1037_17/20260324_1037_40/camera_20260324_1137_f2_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1037_17/20260324_1037_40/camera_20260324_1137_f3_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1037_17/20260324_1042_30/camera_20260324_1142_f1_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1037_17/20260324_1042_30/camera_20260324_1142_f2_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1037_17/20260324_1042_30/camera_20260324_1142_f3_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1048_29/20260324_1049_06/camera_20260324_1149_f1_extracted')
    # record_folder = os.path.join(base_folder, '20260324_1048_29/20260324_1049_06/camera_20260324_1149_f2_extracted')

    # ------------------ 20260325_reshafim ------------------------------ (10" quadrotor target)
    base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/scenarios_vis/20260325_reshafim'
    # record_folder = os.path.join(base_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307_f1_extracted')
    # record_folder = os.path.join(base_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307_f2_extracted')
    # record_folder = os.path.join(base_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307_f3_extracted')
    # record_folder = os.path.join(base_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307_f4_extracted')
    record_folder = os.path.join(base_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307_f5_extracted')



    frame_size = (640, 480)
    frame_resize = None
    start_time = -np.inf
    scen_name = common_utils.path_to_scenario_name(os.path.join(record_folder,'..'))
    # polygons_file = os.path.join(record_folder, scen_name+'_recorded_detection_roi_polygons.yaml')  # ***optional
    polygons_file = os.path.join(record_folder, scen_name+'_manual_detection_roi_polygons.yaml')  # ***optional

    # -------------------- yolov8n -------------
    # model_path = 'runs/detect/drone_detector_yolov8n/weights/best.pt'
    # detection_frame_size = (640, 480)
    # detection_frame_size = (320, 320)
    # detection_frame_size = (256, 256)
    # detection_frame_size = (224, 224)
    # detector_type = 'yolov8n'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov8n/weights/best_320.onnx'
    # detection_frame_size = (320, 320)
    # detector_type = 'yolov8n'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov8n/weights/best_256.onnx'
    # model_path = '/home/roee/Projects/deep_drone_detector/runs/detect/20250709_drone_detector_yolov8n3/weights/best_256.onnx'
    # detection_frame_size = (256, 256)
    # detector_type = 'yolov8n'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov8n/weights/best_224.onnx'
    # detection_frame_size = (224, 224)
    # detector_type = 'yolov8n'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'
    # config_path = None

    # -------------------- yolov11n -------------
    # model_path = 'runs/detect/drone_detector_yolov11n_320x240_20251021/weights/best.pt'
    # detection_frame_size = (320, 320)
    # detector_type = 'yolov11n'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov11n_320x240_20251021/weights/yolov11n_best_lulav_dit_320x320.onnx'
    # detection_frame_size = (320, 320)
    # detector_type = 'yolov11n'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov11n_320x240_20251021/weights/yolov11n_best_lulav_dit_256x256.onnx'
    # detection_frame_size = (256, 256)
    # detector_type = 'yolov11n'
    # config_path = None

    # model_path = 'runs/detect/drone_detector_yolov11n_320x240_20251021/weights/yolov11n_best_lulav_dit_256x256_openvino.xml'
    # config_path = 'runs/detect/drone_detector_yolov11n_320x240_20251021/weights/yolov11n_best_lulav_dit_256x256_openvino.bin'
    # detection_frame_size = (256, 256)
    # detector_type = 'yolov11n'

    # model_path = '/home/roee/Projects/deep_drone_detector/yolo_detector/runs/drone_detector_yolov26n_256x256_20260119/weights/best_256x256_openvino.xml'
    # config_path = '/home/roee/Projects/deep_drone_detector/yolo_detector/runs/drone_detector_yolov26n_256x256_20260119/weights/best_256x256_openvino.bin'
    # detection_frame_size = (256, 256)
    # detector_type = 'yolov26n'
    # detector_name = 'yolov26n_256x256_20260119'

    # model_path = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20260323/models/ultralytics_yolo26_20260324/best_256x256_v2_tmp_openvino.xml'
    # config_path = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20260323/models/ultralytics_yolo26_20260324/best_256x256_v2_tmp_openvino.bin'
    model_path = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20260323/models/ultralytics_yolo26_20260325/yolov26n_256x256_20260325_mixed_p2/best_256x256_openvino.xml'
    config_path = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20260323/models/ultralytics_yolo26_20260325/yolov26n_256x256_20260325_mixed_p2/best_256x256_openvino.bin'

    detection_frame_size = (256, 256)
    detector_type = 'yolov26np2'
    detector_name = 'yolov26n_256x256_20260325'


    # -------------------- nanodet -------------
    # model_path = "/home/roee/Projects/nanodet/workspace/nanodet-plus-m_320/model_best/nanodet_model_best.pth"
    # config_path = "/home/roee/Projects/nanodet/config/nanodet-plus-m_320_lulav_dit.yml"
    # detection_frame_size = (320, 320)  # (w, h)
    # detector_type = 'nanodet-plus-m'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'

    # model_path = "/home/roee/Projects/nanodet/workspace/nanodet-plus-m_320/model_best/nanodet_plus_m_320_lulav_dit_model_best.onnx"
    # config_path = "/home/roee/Projects/nanodet/config/nanodet-plus-m_320_lulav_dit.yml"
    # detection_frame_size = (320, 320)  # (w, h)
    # detector_type = 'nanodet-plus-m'  #  'yolov8n' / 'yolov11n' / 'nanodet-plus-m'

    step_mode = 'detect_only'  # 'detect_only' / 'detect_track' / 'track_only'

    # set video writer
    output_video_file = os.path.join(record_folder, scen_name+'_results_' + detector_name + '.avi')
    if output_video_file is not None:
        print('saving record video to: {}'.format(output_video_file))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_file, fourcc, 20, (640, 480))

    # setup blob tracker
    dttr = detector_tracker.DetectorTracker(model_path, detection_frame_size, detector_type=detector_type,
                                            bbox_roi_intersection_th=0.1, detector_use_cpu=False, verbose=False,
                                            detector_config_path=config_path)


    # get roi polygons per frame
    if polygons_file is not None:
        polygons_per_frame = PolygonPerFrame(frame_resize)
        polygons_per_frame.load(polygons_file)
        if (polygons_per_frame.frame_size[0] != frame_size[0] or
                polygons_per_frame.frame_size[1] != frame_size[1]):
            raise Exception('frame size does not fit polygons_per_frame frame size!')
    else:
        polygons_per_frame = None

    # get record
    record = StandardRecord(record_folder)

    # run on all frames
    tracking_active = False
    track = {'id': None, 'score': None, 'bbox': None}
    max_track_id = 0
    video_initialized = False
    frame_id = 0
    tr = []
    frame_time_step = np.median(np.diff([x['time'] for x in record.frames]))
    polygon_valid_time_gap = frame_time_step * 0.5
    for frame in record.frames:
        if frame['time'] >= start_time:
            img = cv2.imread(frame['image_file'])

            if img is None:
                continue

            if frame_resize is not None:
                img = cv2.resize(img, frame_resize)

            image_size = (img.shape[1], img.shape[0])

            if polygons_per_frame is not None:
                # roi_polygon = polygons_per_frame.get(frame_id)
                roi_polygon = polygons_per_frame.get_time(frame['time'], valid_time_gap=polygon_valid_time_gap)
                if roi_polygon is not None:
                    # dttr.set_detection_roi_polygon(roi_polygon, method='crop', image_size)
                    dttr.set_detection_roi_polygon(roi_polygon, 'hybrid', image_size)

            # test - tmp
            # if frame_id > 50:
            #     roi_polygon = np.zeros((0,2), dtype=np.int32)
            # dttr.set_detection_roi_polygon(roi_polygon, method='crop')

            if (output_video_file is not None) and (not video_initialized):
                print('saving record video to: {}'.format(output_video_file))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_video_file, fourcc, 20, (int(img.shape[1]), int(img.shape[0])))
                video_initialized = True

            # detect and track
            if step_mode == 'detect_only':
                tr = dttr.step(img, conf_threshold=0.4, nms_iou_threshold=0.4, max_num_detections=10,
                               operation_mode='detect_only')
            elif step_mode == 'detect_track':
                tr = dttr.step(img, conf_threshold=0.5, nms_iou_threshold=0.4, max_num_detections=10,
                               operation_mode='detect_track')
            elif step_mode == 'track_only':
                # detect once, and then track only
                if len(tr)>0:
                    tr = dttr.step(img, conf_threshold=0.5, nms_iou_threshold=0.4, max_num_detections=10, operation_mode='track_only')
                else:
                    tr = dttr.step(img, conf_threshold=0.5, nms_iou_threshold=0.4, max_num_detections=10,
                               operation_mode='detect_track')

            img_to_draw = dttr.draw(img)
            img_to_draw = cv2.putText(img_to_draw, '{:d}'.format(frame_id), (20, 20),
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(50, 100,200), thickness=2)
            # draw
            cv2.imshow('Frame', img_to_draw)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if output_video_file is not None:
                out.write(img_to_draw)

            frame_id = frame_id+1


    # Closes all the frames
    cv2.destroyAllWindows()

    if output_video_file is not None:
        out.release()