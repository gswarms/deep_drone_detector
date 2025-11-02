""" Detect and track blobs in image
"""
import cv2
import numpy as np
import cv_core
if cv_core.__version__ != '1.1.1':
    raise Exception('invalid cv_core version {}! must be 1.1.1'.format(cv_core.__version__))
# import time
# import opencv_tracker


class DetectorTracker:
    """
    detect and track blobs in image
    """
    def __init__(self, detector_model_file_path, detection_frame_size, detector_type='yolov8n',
                 tracker_type='klt',
                 bbox_roi_intersection_th=0.1,
                 detector_use_cpu=False, verbose=False,
                 detector_config_path=None):
        """
        Detect and track fixed wing UAV

        :param detector_model_file_path - path to yolo_detector deep learning model
        :param detection_frame_size - (image width, image height) image size for yolo_detector.
                                        To improve runtime, a sub-part of the image aka "detection roi" is passed to the yolo_model.
                                        Image will automatically be resized or cropped to this size!
                                        ** In case we use an .onnx model, this size must fit the onnx predefined size!
        :param bbox_roi_intersection_th - detections with lesser part of their area intersecting with roi polygon will be discarded
        :param detector_use_cpu - force using CPU for yolo_detector even if GPU exists (used for timing tests)
        :param verbose - print log data to screen
        """
        self.detection_frame_size = detection_frame_size
        self.detector_type = detector_type
        self.tracker_type = tracker_type
        self.bbox_roi_intersection_th = bbox_roi_intersection_th
        self.detector_use_cpu = detector_use_cpu
        self.verbose = verbose
        self.tracks = []  # [{'id': <>, 'bbox': (xtl,ytl,w,h), 'score':, <0-100>}, ...]

        #------------------ Setup yolo_detector --------------------
        if self.detector_type in ('yolov8n', 'yolov11n'):
            import yolo_detector
            self.detector = yolo_detector.SingleFrameDetector(detector_model_file_path,
                                                          use_cpu=self.detector_use_cpu, verbose=self.verbose,
                                                              openvino_model_bin_file_path=detector_config_path)
        elif self.detector_type=='nanodet-plus-m':
            import nanodet_detector
            self.detector = nanodet_detector.NanodetDetector(detector_model_file_path, detector_config_path,
                                                              use_cpu=self.detector_use_cpu, verbose=self.verbose)

        # onnx must work with it's predefined input size!
        # if self.detection_frame_size does not match, there might be unexpected behavior!
        # DetectorTracker will crop / resize according to self.detection_frame_size,
        # but then detector will resize to the onnx predefined size.
        # In this case we except to prevent unexpected behavior.
        if self.detector.model_type == 'onnx':
            if (self.detection_frame_size[0] != self.detector.onnx_input_size[0] or
                    self.detection_frame_size[1] != self.detector.onnx_input_size[1]):
                 raise Exception('detection_frame_size ({},{}) does not match onnx predefined size ({},{})'.format(
                     self.detection_frame_size[0], self.detection_frame_size[1],
                     self.detector.onnx_input_size[0], self.detector.onnx_input_size[1]))

        self.detection_roi_polygon = None
        self.detection_roi_bbox = None
        self.detection_roi_method = None
        self.detection_roi_resize_scale = None
        self.detection_frame = np.zeros((self.detection_frame_size[1], self.detection_frame_size[0], 3), dtype=np.uint8)

        # infer once for allocating everything - to reduce first step time
        image = np.zeros((self.detection_frame_size[1], self.detection_frame_size[0], 3), dtype=np.uint8)
        self.detector.detect(image, frame_resize=None)

        #------------------ Setup tracker --------------------

        # # Setup trackers
        # self.max_num_targets = max_num_targets
        # if isinstance(tracker_type, mot.TrackerType):
        #     pass
        # elif isinstance(tracker_type, str) and tracker_type.upper() in mot.TrackerType.__members__:
        #     tracker_type = mot.TrackerType[tracker_type.upper()]
        # else:
        #     raise Exception('invalid tracker_type input type!')
        # if tracker_type == mot.TrackerType.CV2_KLT:
        #     self.trackers = mot.MultiObjectTrackerKLT(self.max_num_targets)
        # else:
        #     self.trackers = mot.MultiObjectTrackerOpencv(self.max_num_targets, tracker_type)
        #
        # # Setup tracks
        # self.track_ids = []
        # self.track_scores = []
        # self.track_count = 0
        # self._max_id = -1
        #
        # other parameters
        # self._tracking_reinit_iou_range = tracking_reinit_iou_range
        # self.overlap_score_threshold = overlap_score_threshold
        # self.max_score = max_score
        # self.blob_center_prediction = None
        # self.detection_valid_mask = np.ones((self.frame_size[1], self.frame_size[0]), dtype=np.uint8)


        return


    def set_detection_roi_polygon(self, polygon_points, method='crop'):
        """
        set yolo_detector ROI from a polygon

        a smaller ROI will be passed for yolo_detector (useful for reducing runtime)

        There are two main metods:
        1. resize: polygon bounding box will be resized to the required size
        2. crop: a box with the required size will be cropped from the image
                 in a way that best overlaps, with the given polygon

        params: polygon_points - (nx2) 2D image points polygon
                                  None - no roi polygon applied - use the entire image.
                                  empty (0X2) - roi is outside the image - don't detect
        params: detection_frame_size - (width, height) frame size for yolo_detector
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
            self.detection_roi_resize_scale = None

        else:
            # get roi polygon
            try:
                polygon_points = np.array(polygon_points).reshape(-1,2)
                if polygon_points.size == 0:
                    # return False
                    polygon_points_valid = False
                else:
                    polygon_points_valid = True
                self.detection_roi_polygon = polygon_points
            except:
                raise Exception('invalid polygon input!')

            # calc roi bbox
            if not polygon_points_valid:
                # detection ROI is empty
                self.detection_roi_bbox = (0, 0, 0, 0)
                self.detection_roi_resize_scale = None

            else:

                if method.lower() == 'crop':
                    """
                    crop required ROI from the image:
                    - if polygon is larger than required roi, take roi with equal margins inside the polygon
                    - if polygon is smaller than required roi, take roi with equal margins arround the polygon
                    each axis is handled separately
                    """
                    self.detection_roi_bbox = self._find_image_bbox_to_crop_from_polygon(self.detection_roi_polygon, self.detection_frame_size, image_size=None)
                    self.detection_roi_resize_scale = None
                    self.detection_roi_method = 'crop'

                elif method.lower() == 'resize':
                    """
                    resize polygon bbox to required ROI:
                    - find polygon bbox
                    - pad bbox to get required ROI scale factor
                    - resize bbox to required ROI size keeping the scale factor
                    """
                    self.detection_roi_bbox, self.detection_roi_resize_scale = self._find_image_bbox_to_resize_from_polygon(
                        self.detection_roi_polygon, self.detection_frame_size, image_size=None)
                    self.detection_roi_method = 'resize'

                elif method.lower() == 'hybrid':
                    """
                    adaptively resize / crop
                    - if polygon is larger than required roi, resize (target is very big, its ok to resize)
                    - if polygon is smaller than required roi, crop (so resolution is not ruined)
                    """
                    # TODO: what if yhe polygon is the full image?
                    dx = max(self.detection_roi_polygon[:, 0]) - min(self.detection_roi_polygon[:, 0])
                    dy = max(self.detection_roi_polygon[:, 1]) - min(self.detection_roi_polygon[:, 1])
                    if dx <= self.detection_frame_size[0] and dy <= self.detection_frame_size[1]:
                        self.detection_roi_bbox = self._find_image_bbox_to_crop_from_polygon(self.detection_roi_polygon,
                                                                                             self.detection_frame_size,
                                                                                             image_size=None)
                        self.detection_roi_resize_scale = None
                        self.detection_roi_method = 'crop'
                    else:
                        self.detection_roi_bbox, self.detection_roi_resize_scale = self._find_image_bbox_to_resize_from_polygon(
                            self.detection_roi_polygon, self.detection_frame_size, image_size=None)
                        self.detection_roi_method = 'resize'

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
        prepare frame for yolo_detector and detect
        1. crop / resize required image ROI
        2. detect using the required ROI
        3. convert yolo_detector coordinates to full image coordinates

        :param: image - (mxn) or (mxnx3) image.
                       * run-time relate Note:
                         if the image is (mxnx3), it will be converted to BGR->HSV->V for the yolo_detector, and passed as is to tracker.
                         if the image is (mxn), it will be passed as is to both yolo_detector and tracker, and avoid any copy.
                         specificly, in the case of KCF tracker, it must use a color image, and therefore will convert the image to BGR
        :param conf_threshold: detections with confidence scores lower than this threshold are discarded.
        :param nms_iou_threshold: The threshold used in NMS to decide whether boxes overlap too much with respect to
                                  Intersection over Union (IoU). If the IoU is greater than this value,
                                  the box with the lower score is suppressed.
        :param max_num_detections: Keeps at most k detections. Useful when you only want the top results.
        """

        if self.detection_roi_method is None:
            # no ROI - use the entire frame
            self.detector.detect(image, frame_resize=None)

        else:
            # make sure roi is in the image
            xtl, ytl, w, h = self.detection_roi_bbox
            xtl = int(np.round(xtl))
            ytl = int(np.round(ytl))
            w = int(np.round(w))
            h = int(np.round(h))
            xtl = min(max(xtl, 0), image.shape[1] - w)
            ytl = min(max(ytl, 0), image.shape[0] - h)
            self.detection_roi_bbox = (xtl, ytl, w, h)
            valid_detection_roi_bbox = w > 0 and h > 0
            if not valid_detection_roi_bbox:
                raise Exception('invalid detection roi bbox!')

            if self.detection_roi_method == 'crop':

                if image.shape[0] < self.detection_frame_size[1] or image.shape[1] < self.detection_frame_size[0]:
                    raise Exception(
                        'image ({}x{}) is smaller than yolo_detector image size ({x})'.format(self.detection_frame_size[1],
                                                                                          self.detection_frame_size[0],
                                                                                          image.shape[1],
                                                                                          image.shape[0]))

                # take roi
                self.detection_frame[:] = image[ytl:ytl+h, xtl:xtl+w]
                # detect without a resize
                results = self.detector.detect(self.detection_frame, frame_resize=None,
                                               conf_threshold=conf_threshold, nms_iou_threshold=nms_iou_threshold)
                # convert back to full image coordinate
                for r in results:
                    r['bbox'] = (r['bbox'][0] + xtl,
                                 r['bbox'][1] + ytl,
                                 r['bbox'][2], r['bbox'][3])

            elif self.detection_roi_method == 'resize':
                # take roi
                self.detection_frame[:] = cv2.resize(image[ytl:ytl + h, xtl:xtl + w], self.detection_frame_size)

                # detect without a resize
                results = self.detector.detect(self.detection_frame, frame_resize=self.detection_frame_size,
                                               conf_threshold=conf_threshold,
                                               nms_iou_threshold=nms_iou_threshold)
                # convert back to full image coordinate
                # TODO: add scale
                for r in results:
                    scale_x, scale_y = self.detection_roi_resize_scale
                    r['bbox'] = (int(np.round(r['bbox'][0] * scale_x + xtl)),
                                 int(np.round(r['bbox'][1] * scale_y + ytl)),
                                 int(np.round(r['bbox'][2] * scale_x)),
                                 int(np.round(r['bbox'][3] * scale_y)))

            else:
                raise Exception('invalid detection_roi_method')

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

    @staticmethod
    def _find_image_bbox_to_crop_from_polygon(polygon_points, required_roi_size, image_size=None):
        """
        find the best bbox to crop from an image, given a polygon and a required roi size
        - if polygon is larger than required roi, take roi with equal margins inside the polygon
        - if polygon is smaller than required roi, take roi with equal margins around the polygon
        each axis is handled separately (e.g. one may be bigger while the other may be smaller)

        param: polygon_points - (nx2) polygon 2D points
        param: required_roi_size - (dx, dy) required ROI size (x-left, y-down).
        param: image_size - (width, height)
                            if None - image borders are not considered
        """

        polygon_points = np.asarray(polygon_points)
        if polygon_points.shape[1] != 2:
            raise Exception('invalid polygon_points shape! should be (nx2)')

        if image_size is not None and (required_roi_size[0] > image_size[0] or required_roi_size[1] > image_size[1]):
            raise Exception('required ROI size is larger than the image size!')

        p_xmn = np.min(polygon_points[:, 0])
        p_xmx = np.max(polygon_points[:, 0])
        p_ymn = np.min(polygon_points[:, 1])
        p_ymx = np.max(polygon_points[:, 1])
        p_w = p_xmx - p_xmn
        p_h = p_ymx - p_ymn

        if p_w <= required_roi_size[0]:
            xtl = int(np.floor(p_xmn - float(required_roi_size[0] - p_w) / 2))
        else:
            xtl = int(np.floor(p_xmn + float(p_w - required_roi_size[0]) / 2))

        if p_h <= required_roi_size[1]:
            ytl = int(np.floor(p_ymn - float(required_roi_size[1] - p_h) / 2))
        else:
            ytl = int(np.floor(p_ymn + float(p_h - required_roi_size[1]) / 2))

        if image_size is not None:
            xtl = min(max(xtl, 0), image_size[0] - required_roi_size[0])
            ytl = min(max(ytl, 0), image_size[1] - required_roi_size[1])

        roi_bbox = (xtl, ytl, required_roi_size[0], required_roi_size[1])

        return roi_bbox

    @staticmethod
    def _find_image_bbox_to_resize_from_polygon(polygon_points, required_roi_size, image_size=None):
        """
        find the best bbox to crop from an image, and then resize given a polygon and a required roi size

        1. calc polygon bbox
        2. pad polygon bbox to fit the required roi scale factor

        param: polygon_points - (nx2) polygon 2D points
        param: required_roi_size - (dx, dy) required ROI size (x-left, y-down).
        param: image_size - (width, height)
                            if None - image borders are not considered
        """

        polygon_points = np.asarray(polygon_points)
        if polygon_points.shape[1] != 2:
            raise Exception('invalid polygon_points shape! should be (nx2)')

        if image_size is not None and (required_roi_size[0] > image_size[0] or required_roi_size[1] > image_size[1]):
            raise Exception('required ROI size is larger than the image size!')

        p_xmn = np.min(polygon_points[:, 0])
        p_xmx = np.max(polygon_points[:, 0])
        p_ymn = np.min(polygon_points[:, 1])
        p_ymx = np.max(polygon_points[:, 1])
        p_w = p_xmx - p_xmn
        p_h = p_ymx - p_ymn

        x_scale = p_w / required_roi_size[0]
        y_scale = p_h / required_roi_size[1]

        if  y_scale > x_scale:  # pad x-axis to reach the desired scale
            p_w_adjusted = required_roi_size[0] * y_scale
            dx = (p_w_adjusted - p_w)/2
            xtl = int(np.floor(p_xmn - dx))
            xbr = int(np.floor(p_xmx + dx))
            roi_bbox = (xtl, p_ymn, xbr-xtl, p_h)

        else:  # pad y-axis to reach the desired scale
            p_h_adjusted = required_roi_size[1] * x_scale
            dy = (p_h_adjusted - p_h) / 2
            ytl = int(np.floor(p_ymn - dy))
            ybr = int(np.floor(p_ymx + dy))
            roi_bbox = (p_xmn, ytl, p_w, ybr - ytl)

        if image_size is not None:
            [xtl, ytl, w, h] = roi_bbox
            xtl = min(max(xtl, 0), image_size[0] - w)
            ytl = min(max(ytl, 0), image_size[1] - h)
            roi_bbox = (xtl, ytl, w, h)

        return roi_bbox, (x_scale, y_scale)


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

        # draw yolo_detector roi
        if self.detection_roi_method is not None:
            poly = np.floor(self.detection_roi_polygon).astype(np.int32)
            img = cv2.polylines(img, [poly], isClosed=True, color=(255, 50, 50), thickness=1)

            start_point = (int(np.floor(self.detection_roi_bbox[0])), int(np.floor(self.detection_roi_bbox[1])))
            end_point = (int(np.floor(self.detection_roi_bbox[0] + self.detection_roi_bbox[2])), int(np.floor(self.detection_roi_bbox[1] + self.detection_roi_bbox[3])))
            img = cv2.rectangle(img, start_point, end_point, color=(255, 50, 50), thickness=1)

        return img
