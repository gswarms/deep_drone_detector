""" bounding box tools
"""
import numpy as np
from enum import Enum


class BboxOverlapScoreType(Enum):
    IOU = 1
    MinOU = 2

# class Bbox:
#     """
#     object for handling image bounding box
#     """
#     def __init__(self, top_left_x, top_left_y, size_x, size_y):
#         self.top_left = (float(top_left_x), float(top_left_y))
#         self.size = (float(size_x), float(size_y))
#         self.area = self.size[0] * self.size[1]
#
#     def intersect(self, other):
#         """
#         calc overlap area between two bounding boxes
#         """
#         if not isinstance(other, Bbox):
#             raise Exception('invalid bounding box!')
#
#         x_tl = max(self.top_left[0], other.top_left[0])
#         y_tl = min(self.top_left[1], other.top_left[1])
#
#         x_br = max(self.top_left[0] + self.size[0], other.top_left[0] + other.size[0])
#         y_br = min(self.top_left[1] + self.size[1], other.top_left[1] + other.size[1])
#
#         dx = max(0, x_br - x_tl)
#         dy = max(0, y_br - y_tl)
#
#         if dx > 0 and dy > 0:
#             res = Bbox(x_tl, y_tl, dx, dy)
#         else:
#             res = None
#
#         return res
#
#     def bbox_overlap_area(self, other):
#         """
#         calc overlap area between two bounding boxes
#         """
#         bbox = self.intersect(other)
#         if bbox is not None:
#             a = bbox.area
#         else:
#             a = 0
#         return a
#
#     def bbox_overlap_iou_score(self, other):
#         """
#         calc overlap area between two bounding boxes
#         """
#         bbox = self.intersect(other)
#         a_union = self.area + other.area - bbox.area
#         iou = bbox.area / a_union
#         return  iou
#
#     def point_in_bbox(self, points):
#         """
#         check if points are in bbox
#         """
#         points = np.array(points)
#         if points.shape[1] != 2:
#             raise Exception('invalid points!')
#         idx1 = np.bitwise_and(points[:, 0] >= self.top_left[0], points[:, 1] >= self.top_left[1])
#         bt = self.top_left + self.size
#         idx2 = np.bitwise_and(points[:, 0] <= bt[0], points[:, 1] <= bt[1])
#         return np.bitwise_and(idx1, idx2)


def bbox_overlap_area(bbox1, bbox2):
    """
    calc overlap area between two bounding boxes

    param: bbox1: (x, y, w, h) follow opencv tracker format
                  (topleft_col, topleft_row, num_cols, num_rows)
    param: bbox2: (x, y, w, h) follow opencv tracker format
                  (topleft_col, topleft_row, num_cols, num_rows)
    """
    x_tl = max(bbox1[1], bbox2[1])
    y_tl = max(bbox1[0], bbox2[0])

    x_br = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    y_br = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])

    dx = max(0, x_br - x_tl)
    dy = max(0, y_br - y_tl)
    return dx * dy


def bbox_overlap_score(bbox1, bbox2, score_type=BboxOverlapScoreType.IOU):
    """
    calc overlap area between two bounding boxes

    param: bbox1: (x, y, w, h) follow opencv tracker format
                  (topleft_col, topleft_row, num_cols, num_rows)
    param: bbox2: (x, y, w, h) follow opencv tracker format
                  (topleft_col, topleft_row, num_cols, num_rows)
    param: score_type: score type BboxOverlapScoreType enum

    """
    if is_bbox_overlap(bbox1, bbox2):

        a1 = float(bbox1[2]) * float(bbox1[3])
        a2 = float(bbox2[2]) * float(bbox2[3])
        a_intersection = bbox_overlap_area(bbox1, bbox2)

        if score_type == BboxOverlapScoreType.IOU:
            a_union = a1 + a2 - a_intersection
            score = a_intersection / a_union

        elif score_type == BboxOverlapScoreType.MinOU:
            score = a_intersection / min(a1, a2)

        else:
            raise Exception('invalid score type!')

    else:
        score = 0

    return  score

def is_bbox_overlap(bbox1, bbox2):
    """
    check two bounding boxes overlap

    param: bbox1: (x, y, w, h) follow opencv tracker format
                  (topleft_col, topleft_row, num_cols, num_rows)
    param: bbox2: (x, y, w, h) follow opencv tracker format
                  (topleft_col, topleft_row, num_cols, num_rows)
    """

    is_not_x = (bbox1[0] + bbox1[2] < bbox2[0]) or (bbox2[0] + bbox2[2] < bbox1[0])
    is_not_y = (bbox1[1] + bbox1[3] < bbox2[1]) or (bbox2[1] + bbox2[3] < bbox1[1])

    return not(is_not_x) and not(is_not_y)

def point_in_bbox(bbox, points):
    """
    check if points are in bbox

    param: bbox: (x, y, w, h) follow opencv tracker format
                  (topleft_col, topleft_row, num_cols, num_rows)
    """
    points = np.array(points)
    if points.shape[1] != 2:
        raise Exception('invalid points!')
    idx1 = np.bitwise_and(points[:, 0] >= bbox[0], points[:, 1] >= bbox[1])
    bt = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    idx2 = np.bitwise_and(points[:, 0] <= bt[0], points[:, 1] <= bt[1])
    return np.bitwise_and(idx1, idx2)

def bbox_in_frame(bbox, image_size, bbox_overlap_ratio, epsilon=1e-8):
    """
    check if bounding box is in the image borders

    param: bbox: (x, y, w, h) follow opencv tracker format
                  (topleft_col, topleft_row, num_cols, num_rows)
    param: image_size: (w, h) in pixels
    param: bbox_overlap_ratio: scalar between 0 and 1.
                       bbox area inside the image
                r =   ----------------------------
                            total bbox area
                if r >= bbox_overlap_ratio bbox is considered inside the image
                if r <  bbox_overlap_ratio bbox is considered outside the image

                if bbox_overlap_ratio is inf, all bbox must be inside the image
                if bbox_overlap_ratio <= 0, any bbox overlap with the image will be considered as in the image
    """

    if not _bbox_valid(bbox):
        raise Exception('invalid bbox!')

    image_bbox = (0, 0, image_size[0], image_size[1])
    intersection_area = bbox_overlap_area(bbox, image_bbox)
    bbox_area = bbox[2] * bbox[3]

    if bbox_overlap_ratio <= 0:
        is_in_frame = intersection_area > 0
    elif bbox_overlap_ratio >= 1:
        is_in_frame = intersection_area == (bbox_area - epsilon)
    else:
        is_in_frame =  bbox_overlap_ratio <= (intersection_area / bbox_area)

    return is_in_frame

def _bbox_valid(bbox):
    """
    check if bbox is valid

     param: bbox: (x, y, w, h) follow opencv tracker format
                  (topleft_col, topleft_row, num_cols, num_rows)
    """

    valid_size = np.array(bbox).size == 4
    bbox_valid = valid_size and bbox[2] >= 0 and bbox[3] >= 0
    return bbox_valid


def points_to_bbox(points):
    """
    find bounding box for a set of 2D points

    param: points: (nx2) set of points (x, y)
    return: bbox: (x, y, w, h)
    """
    xmn = min(points[0, :])
    xmx = max(points[0, :])
    ymn = min(points[1, :])
    ymx = max(points[1, :])
    bbox = (xmn, ymn, xmx-xmn, ymx-ymn)
    return bbox


def match_bbox_sets(bbox1, bbox2, match_iou_th, nms_iou_th=None):
    """
    match bboxs from two sets only by overlap!

    Algorithm:
    1) calc IOU score matrix for each set1-set2 bbox pair
    2) loop over:
       - select best score pair as match (if > iou_match_th)
       - remove the selected pair score list

    None-Maximal Suppression:
        after matches are found, unmatched bboxs with high overlap to matched bboxes are considered invalid
        The list of invalid bboxes1 and invalid bboxes2 is outputed to be used by the user

    params: bbox1 list of n bboxs: [(x1, y1, w1, h1), ... , (xn, yn, wn, hn)]
    params: bbox2 list of m bboxs: [(x1, y1, w1, h1), ... , (xm, ym, wm, hm)]
    params: iou_match_th minimum iou score to be considered a match
    params: nms_iou_match_th minimum iou score to be taken into account in NMS
                             None - don't perform NMS
                             Note: iou_match_th <= nms_iou_match_th

    :return matched_pairs - kx2 array with match indexes (for each match)
    :return bbox1_status, bbox2_status - nx1 and mx1 arrays that stats bbx status
                                         0 - not matched
                                         1 - matched
                                        -1 - filtered by nms
    """

    n1 = len(bbox1)
    n2 = len(bbox2)
    matched_pairs = []
    bbox1_status = np.zeros((n1, 1), dtype=np.int8)
    bbox2_status = np.zeros((n2, 1), dtype=np.int8)
    if n1 == 0 or n2 == 0:
        return matched_pairs, bbox1_status, bbox2_status

    # score matrix
    score_matrix = np.zeros((n1, n2), dtype=np.float32)
    for i in range(n1):
        for j in range(n2):
            score_matrix[i, j] = bbox_overlap_score(bbox1[i], bbox2[j], score_type=BboxOverlapScoreType.IOU)
            # score_matrix[i, j] = bbu.bbox_overlap_score(bbox1[i], bbox2[j], score_type=bbu.BboxOverlapScoreType.MinOU)
    score_matrix[np.where(score_matrix < match_iou_th)] = 0

    # find the best matches
    score_matrix_tmp = score_matrix.copy()
    if n1 > 0 and n2 > 0 and not(np.all(score_matrix_tmp == 0)):
        # print('num detections: {} num tracks: {}'.format(n1, n2))
        matched_bbox1 = np.zeros((n1), dtype=bool)
        matched_bbox2 = np.zeros((n2), dtype=bool)
        while (not np.all(matched_bbox1)) and (not np.all(matched_bbox2)) and (not np.all(score_matrix_tmp == 0)):
            idx = np.argmax(score_matrix_tmp)
            ii, jj = np.unravel_index(idx, (n1, n2), order='C')
            matched_pairs.append({'idx1': ii, 'idx2': jj, 'score': score_matrix_tmp[ii, jj]})
            score_matrix_tmp[ii, :] = 0
            score_matrix_tmp[:, jj] = 0
            matched_bbox1[ii] = True
            matched_bbox2[jj] = True

    for m in matched_pairs:
        bbox1_status[m['idx1']] = 1
        bbox2_status[m['idx2']] = 1

    # non maximal suppression
    if nms_iou_th is not None:

        if nms_iou_th < match_iou_th:
            raise Exception('match_nms_nms_iou_th must be smaller or equal to match_iou_th')

        # find all overlapping bbox pairs
        overlap_matrix = score_matrix >= nms_iou_th
        # remove all actual matches
        for m in matched_pairs:
            overlap_matrix[m['idx1'], m['idx2']] = False

        # get all nms bboxs
        nms_invalid_idx1 = np.any(overlap_matrix, axis=1)
        nms_invalid_idx2 = np.any(overlap_matrix, axis=0)

        bbox1_status[nms_invalid_idx1] = -1
        bbox2_status[nms_invalid_idx2] = -1

    return matched_pairs, bbox1_status, bbox2_status
