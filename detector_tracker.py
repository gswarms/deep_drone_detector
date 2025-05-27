""" Detect and track blobs in image
"""
import time
from pickletools import uint8

import cv2
import numpy as np
# import yaml
# import os
# from ultralytics import YOLO
# from . import bbox_utils as bbu
import cv_core
from triton.language import dtype

import detection


if cv_core.__version__ != '0.1.1':
    raise Exception('invalid cv_core version {}! must be 0.1.1'.format(cv_core.__version__))
# import time



class DetectorTracker:
    """
    detect and track blobs in image
    """
    def __init__(self, detector_model_file_path, detection_frame_size,
                 bbox_roi_intersection_th=0.1,
                 detector_use_cpu=False, verbose=False):
        """
        Detect and track fixed wing UAV

        :param detector_model_file_path - path to detection deep learning model
        :param detection_frame_size - (image width, image height) for detection image
                                       image will automatically be resized or cropped to this size!
        :param bbox_roi_intersection_th - detections with lesser part of their area intersecting with roi polygon will be discarded
        :param detector_use_cpu - force using CPU for detection even if GPU exists (used for timing tests)
        :param verbose - print log data to screen
        """
        self.detection_frame_size = detection_frame_size
        self.bbox_roi_intersection_th = bbox_roi_intersection_th
        self.detector_use_cpu = detector_use_cpu
        self.verbose = verbose

        # Setup detection
        self.detector = detection.SingleFrameDetector(detector_model_file_path,
                                                      use_cpu=self.detector_use_cpu, verbose=self.verbose)


        self.detection_roi_polygon = None
        self.detection_roi_bbox = None
        self.detection_roi_method = None

        self.detection_frame = np.zeros((self.detection_frame_size[1], self.detection_frame_size[0], 3), dtype=np.uint8)
        self.tracks = []  # [{'id': <>, 'bbox': (xtl,ytl,w,h), 'score':, <0-100>}, ...]

        # infer once for allocating everything - to reduce first step time
        image = np.zeros((self.detection_frame_size[1], self.detection_frame_size[0], 3), dtype=np.uint8)
        self.detector.detect(image, frame_resize=self.detection_frame_size)

        return


    def set_detection_roi_polygon(self, polygon_points, method='crop'):
        """
        set detection ROI from a polygon

        a smaller ROI will be passed for detection (useful for reducing runtime)

        There are two main metods:
        1. resize: polygon bounding box will be resized to the required size
        2. crop: a box with the required size will be cropped from the image
                 in a way that best overlaps, with the given polygon

        params: polygon_points - (nx2) 2D image points polygon
        params: detection_frame_size - (width, height) frame size for detection
        params: method - one of the following: ["resize","crop"]
                         crop - crop a part of the image of the required size that best overlaps the polygon
                                if polygon is larget than roi, take equal margins around polygon
                                if polygon is smaller than roi, roi is symmetric to polygon center
                                each axis is set separately
                         resize - take the tight polygon bounding box, and resize it to the required size
        """

        # calc roi bbox
        if polygon_points is None:
            self.detection_roi_polygon = None
            self.detection_roi_bbox = None
            self.detection_roi_method = None

        else:
            # get roi polygon
            try:
                polygon_points = np.array(polygon_points).reshape(-1,2)
                if polygon_points.size == 0:
                    return False
                self.detection_roi_polygon = polygon_points
            except:
                raise Exception('invalid polygon input!')

            if method.lower() in ['crop', 'resize']:
                self.detection_roi_method = method
            else:
                raise Exception('invalid polygon roi method: {}!'.format(method))

            # calc roi bbox
            xmn = np.min(self.detection_roi_polygon[:, 0])
            xmx = np.max(self.detection_roi_polygon[:, 0])
            ymn = np.min(self.detection_roi_polygon[:, 1])
            ymx = np.max(self.detection_roi_polygon[:, 1])
            w = xmx - xmn
            h = ymx - ymn
            xc = np.mean(self.detection_roi_polygon[:, 0])
            yc = np.mean(self.detection_roi_polygon[:, 1])

            if method == 'crop':
                # if polygon is larget than roi, take equal margins around polygon
                # if polygon is smaller than roi, roi is symmetric to polygon center
                # each axis is set separately
                if w <= self.detection_frame_size[0]:
                    xtl = int(np.floor(xc - (float(self.detection_frame_size[0])/2)))
                else:
                    xtl = int(np.floor(xmn + float(w - self.detection_frame_size[0])/2))

                if h <= self.detection_frame_size[1]:
                    ytl = int(np.floor(yc - (float(self.detection_frame_size[1])/2)))
                else:
                    ytl = int(np.floor(ymn + float(h - self.detection_frame_size[1])/2))

                self.detection_roi_bbox = (xtl, ytl, self.detection_frame_size[0], self.detection_frame_size[1])

            elif method == 'resize':
                self.detection_roi_bbox = (xmn, ymn, xmx - xmn, ymx - ymn)

            else:
                raise Exception('invalid roi method {}!'.format(self.detection_roi_method))

        return True

    def get_tracks(self):
        """
        get all current tracks
        """
        tracks = []
        for i, tr in enumerate(self.trackers.tracks):
            tracks.append({'id': self.track_ids[i], 'score': self.track_scores[i], 'bbox': tr['bbox']})
        return tracks

    def step(self, image, conf_threshold=0.4, nms_iou_threshold=0.5, max_num_detections=10):
        """
        prepare frame for detection and detect
        1. crop / resize required image ROI
        2. detect using the required ROI
        3. convert detection coordinates to full image coordinates

        :param: image - (mxn) or (mxnx3) image.
                       * run-time relate Note:
                         if the image is (mxnx3), it will be converted to BGR->HSV->V for the detection, and passed as is to tracker.
                         if the image is (mxn), it will be passed as is to both detection and tracker, and avoid any copy.
                         specificly, in the case of KCF tracker, it must use a color image, and therefore will convert the image to BGR
        :param conf_threshold: detections with confidence scores lower than this threshold are discarded.
        :param nms_iou_threshold: The threshold used in NMS to decide whether boxes overlap too much with respect to
                                  Intersection over Union (IoU). If the IoU is greater than this value,
                                  the box with the lower score is suppressed.
        :param max_num_detections: Keeps at most k detections. Useful when you only want the top results.
        """


        if self.detection_roi_method is None:
            self.detector.detect(image, frame_resize=None)

        else:

            if self.detection_roi_method == 'crop':

                if image.shape[0] < self.detection_frame_size[1] or image.shape[1] < self.detection_frame_size[0]:
                    raise Exception(
                        'image ({}x{}) is smaller than detection image size ({x})'.format(self.detection_frame_size[1],
                                                                                          self.detection_frame_size[0],
                                                                                          image.shape[1],
                                                                                          image.shape[0]))
                # make sure roi is in the image
                xtl, ytl, w, h = self.detection_roi_bbox
                xtl = min(max(xtl, 0), image.shape[1]-w)
                ytl = min(max(ytl, 0), image.shape[0]-h)

                # take roi
                self.detection_frame[:] = image[ytl:ytl+h, xtl:xtl+w]

                # detect without a resize
                results = self.detector.detect(self.detection_frame, frame_resize=None,
                                               conf_threshold=conf_threshold, nms_iou_threshold=nms_iou_threshold)


                # convert back to full image coordinate
                for r in results:
                    r['bbox'] = (r['bbox'][0] + self.detection_roi_bbox[0],
                                 r['bbox'][1] + self.detection_roi_bbox[1],
                                 r['bbox'][2], r['bbox'][3])

                # discard detections of the roi polygon
                valid_results = []
                for r in results:
                    bbox_intersects = self._bbox_in_detection_polygon(r['bbox'])
                    if bbox_intersects:
                        valid_results.append(r)

                # update self.tracks
                self.tracks = []
                for i, bbox in enumerate(valid_results):
                    self.tracks.append({'id': i, 'score': bbox['confidence'], 'bbox': bbox['bbox']})


            elif self.detection_roi_method == 'resize':
                # make sure roi is inside the image
                xtl, ytl, w, h = self.detection_roi_bbox
                xbr = min(xtl + w, image.shape[1])
                ybr = min(ytl + h, image.shape[0])
                xtl = max(xtl, 0)
                ytl = max(ytl, 0)

                # take roi
                self.detection_frame[:] = image[ytl:ybr, xtl:xbr]

                # resize and detect
                res = self.detector.detect(self.detection_frame, frame_resize=self.detection_frame_size)

                # convert back to full image coordinates
                # update self.tracks
                a = 5

        return self.tracks

    def _bbox_in_detection_polygon(self, bbox):

        bbox = [int(np.floor(v)) for v in bbox]
        bbox_points = np.array([[bbox[0], bbox[1]],
                                [bbox[0] + bbox[2], bbox[1]],
                                [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                                [bbox[0], bbox[1] + bbox[3]]], dtype=np.int32)
        poly = np.floor(self.detection_roi_polygon).astype(np.int32)
        retval, intersected_polygon = cv2.intersectConvexConvex(bbox_points, poly)
        bbx_intersects = False
        if retval > 0:
            bbox_area = bbox[2] * bbox[3]
            if retval >= bbox_area * self.bbox_roi_intersection_th:
                bbx_intersects = True
        return bbx_intersects

    def draw(self, img):
        """
        draw on image
        """
        for tr in self.tracks:

            # draw tracks
            x, y, w, h = [int(v) for v in tr['bbox']]
            img = cv2.rectangle(img, (x, y), (x+w, y+h), color=(100, 255, 255), thickness=1)

            track_id = tr['id']
            track_score = tr['score']
            img = cv2.putText(img, '{}:{:.2f}'.format(track_id, track_score),
                        (x + w, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(100, 255, 255),
                        thickness=1)

        # draw detection roi
        if self.detection_roi_method is not None:
            poly = np.floor(self.detection_roi_polygon).astype(np.int32)
            img = cv2.polylines(img, [poly], isClosed=True, color=(255, 50, 50), thickness=1)

            start_point = (int(np.floor(self.detection_roi_bbox[0])), int(np.floor(self.detection_roi_bbox[1])))
            end_point = (int(np.floor(self.detection_roi_bbox[0] + self.detection_roi_bbox[2])), int(np.floor(self.detection_roi_bbox[1] + self.detection_roi_bbox[3])))
            img = cv2.rectangle(img, start_point, end_point, color=(255, 50, 50), thickness=1)

        return img
