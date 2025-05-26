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

        a1 = bbox1[2] * bbox1[3]
        a2 = bbox2[2] * bbox2[3]
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