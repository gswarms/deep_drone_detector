import os
import re


def path_to_scenario_name(scenario_folder_path):
    """
    reformat a file / folder name to a scenario name.
    path format:  <path>/year_month_day-hour_min_sec_<any suffix>
    scenario name: yyyymmdd_HHMMSS

    :param scenario_folder_path:
    :return:
    """
    base_name = os.path.basename(os.path.abspath(scenario_folder_path))
    sp = re.split('_|-', base_name)
    if len(sp) == 6 and [len(x)for x in sp] == [4, 2, 2, 2, 2, 2]:  # folder naming format yyyy-mm-dd_HH-MM-SS
        scenario_name = '{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(int(sp[0]), int(sp[1]), int(sp[2]), int(sp[3]), int(sp[4]), int(sp[5]))
    elif len(sp) == 2 and [len(x)for x in sp] == [8, 6]:  # folder naming format yyyymmdd_HHMMSS
        scenario_name = '{:08d}_{:06d}'.format(int(sp[0]), int(sp[1]))
    elif len(sp) == 3 and [len(x)for x in sp] == [8, 4, 2]:  # folder naming format yyyymmdd_HHMM_SS
        scenario_name = '{:08d}_{:06d}'.format(int(sp[0]), int(sp[1] + sp[2]))
    else:
        raise Exception('invalid folder naming format: {}'.format(scenario_folder_path))
    return scenario_name
