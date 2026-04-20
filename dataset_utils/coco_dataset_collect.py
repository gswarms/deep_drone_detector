import os
from pathlib import Path
import yaml
import coco_dataset_manager


def merge_dataset(dataset_collect_file, merged_dataset_root_folder):
    """
    merge single experiment scenario datasets to one big dataset
    """

    dataset_collect_file = Path(dataset_collect_file)
    merged_dataset_root_folder = Path(merged_dataset_root_folder)

    if not dataset_collect_file.exists():
        raise Exception('{} does not exist'.format(dataset_collect_file))
    os.makedirs(merged_dataset_root_folder, exist_ok=True)

    try:
        with open(dataset_collect_file, 'r') as f:
            data = yaml.safe_load(f)
    except:
        raise Exception('could not open {}'.format(dataset_collect_file))

    clusters_base_folder = Path(data['cluster_base_folder'])
    if not clusters_base_folder.exists():
        raise Exception('{} base folder not exist'.format(clusters_base_folder))
    cluster_folders = [(clusters_base_folder/Path(x)).resolve() for x in data['cluster_folders']]
    discard_scenario_folders = [(clusters_base_folder/Path(x)).resolve() for x in data['discard_scenario_folders']]

    # init merged dataset
    print('merging dataset to: {}'.format(merged_dataset_root_folder))
    if (merged_dataset_root_folder/'images').exists() or (merged_dataset_root_folder/'annotations').exists():
        raise Exception('merged folder already exists: {}'.format(merged_dataset_root_folder))
    dataset = coco_dataset_manager.CocoDatasetManager()
    dataset.set_root(merged_dataset_root_folder)

    # log
    log_file = merged_dataset_root_folder/'dataset_log.txt'
    log = MergeLogger(log_file)

    scen_count = 0
    n_images = []
    n_annotations = []
    for cluster_folder in cluster_folders:
        if not cluster_folder.exists():
            raise Exception('{} cluster folder does not exist'.format(cluster_folder))
        annotation_folders = [p for p in cluster_folder.rglob("**/annotations") if p.is_dir()]
        for annotation_folder in annotation_folders:
            scen_folder = annotation_folder.parent
            if scen_folder in discard_scenario_folders:
                log.write('discarded scenario: {}'.format(scen_folder.relative_to(clusters_base_folder)))
                continue
            rel_scen_folder = scen_folder.relative_to(clusters_base_folder)
            log.write('scenario: {}'.format(rel_scen_folder))

            # load scene dataset
            scene_json = annotation_folder/'coco_dataset.json'
            scene_dataset = coco_dataset_manager.CocoDatasetManager()
            scene_dataset.load_coco(scene_json, verify_image_files=False)

            # removed_image_ids = scene_dataset.remove_missing_images()
            # print('removed {} missing images!'.format(len(removed_image_ids)))
            # scene_dataset.save_coco(overwrite=True)

            n_images.append(scene_dataset.df_images.shape[0])
            n_annotations.append(scene_dataset.df_annotations.shape[0])

            log.write('{} images, {} annotations'.format(n_images[-1], n_annotations[-1]))

            # merge to big dataset
            dataset.merge_dataset(scene_dataset, merge_hash_duplicates=False,
                      merge_overlapping_bbox_annotations=True, verify_other_images=False, bbox_overlap_iou_th = 0.2,
                      verbose=False)
            scen_count = scen_count + 1

    # save dataset
    dataset.save_coco(dataset_root_folder=None, json_file_name='coco_dataset.json', copy_images=True, overwrite=False)

    img_ids = dataset.get_image_ids()
    log.write('----------------------------------')
    log.write('merged_dataset:')
    log.write('{} scenarios'.format(scen_count))
    log.write('{} images'.format(len(img_ids)))
    log.write('{} annotations'.format(dataset.df_annotations.shape[0]))

class MergeLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.log = open(log_file, 'w')

    def write(self, log_str, log=True, prnt=True):
        if prnt:
            print(log_str)
        if log:
            self.log.write(log_str + '\n')
    def __del__(self):
        self.log.close()



if __name__ == '__main__':
    # -------- dataset_20251211 -------
    base_folder = '/home/roee/Projects/datasets/interceptor_drone/uav_detection_dataset/'
    dataset_collect_file = os.path.join(base_folder, 'dataset_20251211/dataset_scenarios.yaml')
    merged_dataset_root_folder =  os.path.join(base_folder, 'dataset_20251211/merged_dataset_raw')

    # -------- dataset_20260323 -------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/uav_detection_dataset/'
    # dataset_collect_file = os.path.join(base_folder, 'dataset_20260330/dataset_scenarios.yaml')
    # merged_dataset_root_folder =  os.path.join(base_folder, 'dataset_20260330/merged_dataset_raw')

    merge_dataset(dataset_collect_file, merged_dataset_root_folder)

    print('Done!')