import copy

import numpy as np
import cv2
from fontTools.ttLib.woff2 import woff2FlagsSize


def augment_crop_image(img, crop_size, bboxes=None, resize_to_original=False, min_valid_bbox_crop_ratio=0.5):
    """
    augment image by cropping a part of it, and resizing to original image size
    adjust bbox coordinates accordingly

    :param img - [nxm] image
    :param crop_size - (w, h) size of the cropped image
    :param bboxes - list of bounding boxes. Each of the format (xtl, ytl, w, h)
    :param resize_to_original - boolean flag.
                                True - resize cropped image to original image size.
                                False - keep cropped image size.
    :param min_valid_bbox_crop_ratio -
    :return:
    """
    h = img.shape[0]
    w = img.shape[1]

    w_cropped = crop_size[0]
    h_cropped = crop_size[1]

    if w_cropped >= w or h_cropped >= h:
        raise Exception('invalid augment-crop size! cropped size ({},{}) does not fit in image original size ({},{})'.format(w_cropped, h_cropped, w, h))
    if (bboxes is not None) and (not isinstance(bboxes, list)):
        raise Exception('bbox must be a list of lists/tuples (xtl, ytl, w, h)')

    xtl = np.random.randint(0, w - w_cropped)
    ytl = np.random.randint(0, h - h_cropped)
    img_cropped = img[ytl: ytl + h_cropped, xtl: xtl + w_cropped, :]

    if bboxes is None:
        bboxes_cropped = None
    else:
        cropped_img_borders = (xtl, ytl, xtl + w_cropped, ytl + h_cropped)
        bboxes_cropped = []
        for bbox in bboxes:
            bbox_cropped = intersect_bbox(bbox, cropped_img_borders)

            bbox_cropped_valid = False
            if bbox_cropped is not None:
                valid_bbox_crop_ratio = (bbox_cropped[2] * bbox_cropped[3]) / (bbox[2] * bbox[3])
                if valid_bbox_crop_ratio >= min_valid_bbox_crop_ratio:
                    bbox_cropped_valid = True

            if bbox_cropped_valid:
                bboxes_cropped.append((bbox_cropped[0] - xtl, bbox_cropped[1] - ytl, bbox_cropped[2], bbox_cropped[3]))
            else:
                bboxes_cropped.append(None)
                # if bbox_cropped is not None:
                #     print(valid_bbox_crop_ratio)
                #     img_cropped2 = copy.deepcopy(img_cropped)
                #     img_cropped2 = cv2.rectangle(img_cropped2, (int(bbox_cropped[0]-xtl), int(bbox_cropped[1]-ytl)), (int(bbox_cropped[0]+bbox_cropped[2]-xtl), int(bbox_cropped[1]+bbox_cropped[3]-ytl)), (0, 255, 0), 2)
                #     cv2.imshow('tmp1', img_cropped2)
                #     cv2.waitKey(100)
                #     aa=5

    if resize_to_original:
        img_cropped = cv2.resize(img_cropped, (w, h))
        scale_x = w / w_cropped
        scale_y = h / h_cropped
        if bboxes_cropped is None:
            bboxes_cropped = None
        else:
            for i, bbox in enumerate(bboxes_cropped):
                if bbox is not None:
                    bboxes_cropped[i] = (int(np.round(bbox[0] * scale_x)),
                            int(np.round(bbox[1] * scale_y)),
                            int(np.floor(bbox[2] * scale_x)),
                            int(np.floor(bbox[3] * scale_y)))

    return img_cropped, bboxes_cropped


def intersect_bbox(bbox1, bbox2):
    """
    intersect two bboxes

    bbox1 - (xtl, ytl, w, h)
    bbox2 - (xtl, ytl, w, h)

    :return: intersection bbox
             if no valid intersection - return None
    """

    if bbox1 is None or bbox2 is None:
        return None

    x1_min, y1_min, w1, h1 = bbox1
    x2_min, y2_min, w2, h2 = bbox2
    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        intersection_bbox = (inter_x_min, inter_y_min, inter_x_max - inter_x_min, inter_y_max - inter_y_min)
    else:
        intersection_bbox = None  # No intersection

    return intersection_bbox


if __name__ == '__main__':
    img_file = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/images/train/20250402_112645_1743593214885744346.jpg'
    bbox = (345, 384, 45, 15)  # (xtl, ytl, w, h)
    crop_size = (320, 240)

    img = cv2.imread(img_file)
    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

    cv2.imshow('original image', img)
    cv2.waitKey(50)

    for i in range(20):
        cropped_img, cropped_bbox = augment_crop_image(img, crop_size, bboxes=[bbox], resize_to_original=True, min_valid_bbox_crop_ratio=0.4)

        if cropped_bbox[0] is not None:
            x1 = cropped_bbox[0][0]
            y1 = cropped_bbox[0][1]
            x2 = cropped_bbox[0][0] + cropped_bbox[0][2]
            y2 = cropped_bbox[0][1] + cropped_bbox[0][3]
            cropped_img = cv2.rectangle(cropped_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('augmented image', cropped_img)
        cv2.waitKey(100)
        aa=5

    print('Done!')