from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import coco_dataset_manager


def analyse_dataset(dataset):
    """
    analyse dataset

    :param dataset: coco manager
    :return:
    """

    # TODO: display bbox size histogram
    # TODO: display category histogram
    # TODO: display images metadata histogram
    # TODO: display annotations metadata histogram

    # general info
    print('dataset root folder: {}'.format(dataset.root_folder))
    print('json_file: {}'.format(dataset.json_path))
    print('{} images'.format(dataset.df_images.shape[0]))
    print('{} annotations'.format(dataset.df_annotations.shape[0]))

    # category histogram
    h = dataset.get_category_histogram()
    names = [h[x]['name'] for x in h]
    counts = [h[x]['count'] for x in h]
    fig1, ax1 = plt.subplots()
    ax1.set_title("Category Histogram")
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Count")
    ax1.bar(names, counts)
    plt.show(block=False)

    # bbox size histogram
    areas = dataset.get_annotations_area()
    bin_edges = np.linspace(0, max(areas), 1000)
    fig2, ax2 = plt.subplots()
    ax2.hist(areas, bins=bin_edges, edgecolor='black')
    ax2.set_title("Bbox Area Histogram")
    ax2.set_xlabel("bbox area")
    ax2.set_ylabel("Count")
    plt.show(block=False)

    # # metadata histogram
    # areas = dataset.get_images_metadata_histogram('lighting')
    # bin_edges = np.linspace(0, max(areas), 1000)
    # fig2, ax2 = plt.subplots()
    # ax2.hist(areas, bins=bin_edges, edgecolor='black')
    # ax2.set_title("Bbox Area Histogram")
    # ax2.set_xlabel("bbox area")
    # ax2.set_ylabel("Count")
    # plt.show(block=False)


if __name__ == '__main__':

    dataset_path = Path('/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/merged_dataset_raw')
    json_path = dataset_path / 'annotations' / 'coco_dataset.json'

    dataset = coco_dataset_manager.CocoDatasetManager()
    dataset.load_coco(json_path)

    analyse_dataset(dataset)
    print('Done!')
