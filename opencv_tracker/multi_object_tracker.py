import cv2
import cv2.legacy
import numpy as np
# import time
from enum import Enum
from abc import ABC, abstractmethod

class TrackerType(Enum):
    CV2_KCF = 0  # BGR image only!
    CV2_CSRT = 1
    CV2_MedianFlow = 2
    CV2_MOSSE = 3
    CV2_KLT = 4



class BaseMultiObjectTracker(ABC):
    def __init__(self, max_num_tracks, tracker_type:TrackerType):
        """
        This is just a base class to define multi object tracker interface
        """
        self.max_num_trackers = max_num_tracks
        self.tracker_type = tracker_type
        self.tracks = []  # list of tracks
                          # each track is a dict:
                          # {'bbox':(xtl, ytl, w, h), 'tracking_point': (x, y), 'state':<1/0>}
                          #                                                         state is by opencv notation:
                          #                                                         0 - idle / lost
                          #                                                         1 - active

    @abstractmethod
    def initialize(self, frame: np.ndarray, bboxes)->bool:
        """
        Initialize the trackers with the provided frame and bounding boxes.
        This can be used both for first time initialization, and for adding new trackers

        :param frame: The first frame of the video or image to initialize the trackers.
        :param bboxes: List of bounding boxes [(x, y, w, h)] for each object.
        """
        pass  # class specific implementation
        return True

    @abstractmethod
    def update(self, frame: np.ndarray):
        """
        Update all the trackers and get the updated bounding boxes.

        :param frame: The frame to update the trackers.
        :return: A list of updated bounding boxes [(x, y, w, h)].
        """
        pass  # class specific implementation
        return self.tracks

    @abstractmethod
    def reinit_track(self, frame, index, bbox):
        """
        re-initialize tracker with new bbox

        :param frame: The current frame for initializing the new tracker.
        :param index: tracker index to reinit.
        :param bbox: The bounding box (x, y, w, h) of the new object.
        """
        pass  # class specific implementation
        return

    def clear_lost_tracks(self):
        """
        remove lost tracks
        """
        removed_indices = []
        for i in range(len(self.tracks)-1 , -1, -1):
            if self.tracks[i]['state'] is 0:
                self.remove_track(i)
                removed_indices.append(i)
        return removed_indices

    def remove_track(self, index):
        """
        Remove a tracker at a specific index.

        :param index: The index of the tracker to remove.
        """
        if 0 <= index < len(self.tracks):
            self.tracks.pop(index)
        else:
            print(f"Invalid index {index}, cannot remove track.")
        return

    def _bbox_center_point(self, bbox):
        return (int(np.round(bbox[0] + bbox[2] / 2)), int(np.round(bbox[1] + bbox[3] / 2)))

    def draw_tracks(self, frame, color=(100, 255, 255), thickness=1):
        """
        Draw bounding boxes on the frame for all tracked objects.

        :param frame: The frame to draw the bounding boxes on.
        :param color: bbox line color - cv2.rectangle format.
        :param thickness: bbox line thickness - cv2.rectangle format.
        :return: The frame with drawn bounding boxes.
        """

        for tr in self.tracks:
            if tr['bbox'] is not None:
                x, y, w, h = [int(v) for v in tr['bbox']]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                # cv2.putText(frame, '{}:{}'.format(self.track_ids[i],self.track_scores[i]),
                #             (x + w, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)

        return frame


class MultiObjectTrackerOpencv(BaseMultiObjectTracker):
    def __init__(self, max_num_tracks, tracker_type:TrackerType=TrackerType.CV2_CSRT):
        super().__init__(max_num_tracks, tracker_type)  # Call base class constructor
        # Initialize the list to hold individual tracker instances
        self.trackers = []

    def initialize(self, frame, bboxes):
        """
        Initialize the trackers with the provided frame and bounding boxes.
        This can be used both for first time initialization, and for adding new trackers

        :param frame: The first frame of the video or image to initialize the trackers.
        :param bboxes: List of bounding boxes [(x, y, w, h)] for each object.
        """
        for bbox in bboxes:
            if len(self.tracks) < self.max_num_trackers:
                # create opencv tracker
                if self.tracker_type == TrackerType.CV2_CSRT:
                    tracker = cv2.legacy.TrackerCSRT.create()
                elif self.tracker_type == TrackerType.CV2_MOSSE:
                    tracker = cv2.legacy.TrackerMOSSE.create()
                elif self.tracker_type == TrackerType.CV2_KCF:
                    tracker = cv2.legacy.TrackerKCF.create()
                elif self.tracker_type == TrackerType.CV2_MedianFlow:
                    tracker = cv2.legacy.TrackerMedianFlow.create()
                else:
                    raise Exception('invalid racker type!')

                # prepare frame (for KCF - only color image!)
                if self.tracker_type == TrackerType.CV2_KCF and len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # init opencv tracker
                tracker.init(frame, tuple(bbox))

                # add track
                self.trackers.append(tracker)
                self.tracks.append({'bbox': bbox, 'tracking_point': self._bbox_center_point(bbox), 'state': 1})

            else:
                print('max number of trackers {} is reached! not initializing new trackers!'.format(self.max_num_trackers))
        return

    def update(self, frame):
        """
        Update all the trackers and get the updated bounding boxes.

        :param frame: The frame to update the trackers.
        :return: A list of updated bounding boxes [(x, y, w, h)].
        """
        # t1 = time.monotonic()

        # prepare frame (for KCF - only color image!)
        if self.tracker_type == TrackerType.CV2_KCF and len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # update opencv trackers
        for i, tracker in enumerate(self.trackers):
            success, bbox = tracker.update(frame)
            if success:
                self.tracks[i] = {'bbox': bbox, 'tracking_point': self._bbox_center_point(bbox), 'state': 1}
            else:
                self.tracks[i] = {'bbox': None, 'tracking_point': None, 'state': 0}
        # t2 = time.monotonic()
        # print('tracking time: {}'.format(t2-t1))

        return self.tracks

    def reinit_track(self, frame, index, bbox):
        """
        re-initialize tracker with new bbox

        :param frame: The current frame for initializing the new tracker.
        :param index: tracker index to reinit.
        :param bbox: The bounding box (x, y, w, h) of the new object.
        """
        bbox = tuple(bbox)
        if 0 <= index < len(self.tracks):
            self.trackers[index].init(frame, tuple(bbox))
            self.tracks[index] = {'bbox': bbox, 'tracking_point': self._bbox_center_point(bbox), 'state': 1}
        else:
            print(f"Invalid index {index}, cannot remove tracker.")
        return

    def remove_track(self, index):
        """
        Remove a tracker at a specific index.

        :param index: The index of the tracker to remove.
        """
        if 0 <= index < len(self.tracks):
            self.tracks.pop(index)
            self.trackers.pop(index)
        else:
            print(f"Invalid index {index}, cannot remove track.")
        return


class MultiObjectTrackerKLT(BaseMultiObjectTracker):
    def __init__(self, max_num_tracks):
        super().__init__(max_num_tracks, TrackerType.CV2_KLT)  # Call base class constructor

        # Initialize the list to hold individual tracker instances
        self.tracks = []  # list of tracks
                          # each track is a dict:
                          # {'bbox':(xtl, ytl, w, h), 'tracking_point': (x, y), 'state':<1/0>}
                          #                                                         state is by opencv notation:
                          #                                                         0 - idle / lost
                          #                                                         1 - active

        self.max_num_tracks = max_num_tracks
        self.prev_frame = None
        self.klt_default_win_size = 9
        self.klt_maxLevel = 3
        self.klt_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self._klt_feature_roi_mask = None
        self._klt_win_size = None


    def initialize(self, frame, bboxes, init_frame=True, method='gftt'):
        """
        Initialize the trackers with the provided frame and bounding boxes.
        This can be used both for first time initialization, and for adding new trackers

        :param frame: The first frame of the video or image to initialize the trackers.
        :param bboxes: List of bounding boxes [(x, y, w, h)] for each object.
        :param init_frame: boolean that status weather to update frame or not
                           False is relevant only if you already updated the current frame, and want to init new tracks
        :param method: method for selecting tracking point
                       - 'center': just use center of bbox
                       - 'gftt': use opencv good features to track
        """

        if init_frame:
            if len(frame.shape) == 2:
                self.prev_frame = frame
            elif len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.prev_frame = gray
            else:
                raise Exception('invalid image size! Expecting (mxnx2) OR (mxnx3)')

        for bbox in bboxes:
            if len(self.tracks) < self.max_num_tracks:

                if method == 'center':
                    cp = self._bbox_center_point(bbox)

                elif method == 'gftt':
                    # find good features to track in the bbox
                    if self._klt_feature_roi_mask is None:
                        self._klt_feature_roi_mask = np.zeros((self.prev_frame.shape[0],self.prev_frame.shape[1]), dtype=np.uint8)
                    self._klt_feature_roi_mask[:] = 0
                    self._klt_feature_roi_mask[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]] = 1
                    cp = cv2.goodFeaturesToTrack(self.prev_frame, maxCorners=10,  qualityLevel=0.1,
                            minDistance = 1, mask= self._klt_feature_roi_mask)
                    if cp is None:
                        cp = self._bbox_center_point(bbox)
                    else:
                        cp = (cp[0,0,0], cp[0,0,1])
                else:
                    raise Exception('invalid tracking point selection method!')

                self.tracks.append({'bbox': bbox, 'tracking_point':cp, 'state':1})
            else:
                print('max number of tracks {} is reached! not initializing new trackers!'.format(self.max_num_tracks))

        return

    def update(self, frame):
        """
        Update all the trackers and get the updated bounding boxes.

        Note: this will update tracks bbox
              frame will also be updated even id there are no tracks

        :param frame: The frame to update the trackers.
        :return: A list of updated bounding boxes [(x, y, w, h)].
        """

        if self.prev_frame is None and len(self.tracks)>0:
            raise Exception('update before initialize is invalid!')

        # t1 = time.monotonic()
        pts = [tp['tracking_point'] for tp in self.tracks]
        # prev_pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) > 0:
            prev_pts = np.vstack(pts).astype(np.float32)
        else:
            prev_pts = np.zeros((0, 2), dtype=np.float32)

        # get window size from bbox
        # must be the same window size for all points!
        # TODO: we need to support various bbox sizes for multiple targets!
        if len(self.tracks)>0:
            bbox_size = [max((tr['bbox'][2], tr['bbox'][3])) for tr in self.tracks]
            ws = max(self.klt_default_win_size, int(min(bbox_size)))
            self._klt_win_size = (ws, ws)

        next_pts = []
        status = []
        if len(frame.shape) == 2:
            if len(self.tracks) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame, prev_pts, None,
                                                               winSize=self._klt_win_size, maxLevel=self.klt_maxLevel, criteria=self.klt_criteria)
            self.prev_frame = frame.copy()

        elif len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(self.tracks) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, prev_pts, None,
                                                winSize=self._klt_win_size, maxLevel=self.klt_maxLevel, criteria=self.klt_criteria)
            self.prev_frame = gray.copy()

        else:
            raise Exception('invalid image size! Expecting (mxnx2) OR (mxnx3)')

        # update tracks, and Filter out lost tracks
        for i, (pt, st) in enumerate(zip(next_pts, status)):
            if st[0] == 1:
                self.tracks[i]['tracking_point'] = pt
                self.tracks[i]['state'] = st
                xtl = pt[0] - self.tracks[i]['bbox'][2] / 2
                ytl = pt[1] - self.tracks[i]['bbox'][3] / 2
                self.tracks[i]['bbox'] = (xtl, ytl, self.tracks[i]['bbox'][2], self.tracks[i]['bbox'][3])
            else:
                self.tracks[i]['tracking_point'] = None
                self.tracks[i]['state'] = 0
                self.tracks[i]['bbox'] = None

        return self.tracks

    def reinit_track(self, frame, index, bbox):
        """
        re-initializes trackers with new bbox
        does not change previous frame!

        :param frame: not used. just keeping the same interface.
        :param index: tracker index to reinit.
        :param bbox: The bounding box (x, y, w, h) of the new object.
        """

        if index < len(self.tracks):
            self.tracks[index]['bbox'] = bbox
            self.tracks[index]['tracking_point'] = self._bbox_center_point(bbox)
            res = True
        else:
            res = False

        return res

if __name__ == "__main__":
    cap = cv2.VideoCapture('video.mp4')  # Open a video file or webcam

    multi_tracker = MultiObjectTrackerOpencv()

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        exit()

    # Initialize the trackers with bounding boxes (e.g., manually or from another source)
    bboxes = [(100, 100, 50, 50), (300, 200, 60, 60)]  # Example bounding boxes
    multi_tracker.initialize(frame, bboxes)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker with the new frame
        updated_bboxes = multi_tracker.update(frame)

        # Draw the bounding boxes on the frame
        frame_with_tracks = multi_tracker.draw_tracks(frame)

        # Display the result
        cv2.imshow("Multi Object Tracking", frame_with_tracks)

        # Example of adding a new tracker after 200 frames
        if cv2.waitKey(1) & 0xFF == ord('a'):  # Press 'a' to add a new tracker
            new_bbox = (150, 150, 40, 40)  # New bounding box for the new object
            multi_tracker.add_tracker(frame, new_bbox)

        # Example of removing a tracker after 300 frames
        if cv2.waitKey(1) & 0xFF == ord('r'):  # Press 'r' to remove the last tracker
            multi_tracker.remove_tracker(len(multi_tracker.trackers) - 1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()