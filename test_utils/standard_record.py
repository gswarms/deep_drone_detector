"""Extract information from a rosbag. """
import os
import cv2
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


class StandardRecord:
    """
    This object handles LULAV interceptor drone standard folder record

    Basic functionality
    1) get camera frames along with additional data (e.g. bbox reference)
    2) draw recorded frames (and save video)
    """

    def __init__(self, record_folder):
        self.record_folder = record_folder
        if not os.path.isdir(self.record_folder):
            raise Exception('record folder: {} not found!'.format(self.record_folder))
        self.frames = None
        self._get_camera_frames()
        self._get_bbox_ref()

    def _get_camera_frames(self):
        """
        get camera frame times and corresponding image files
        """

        timestamps_file = os.path.join(self.record_folder, 'timestamps.txt')
        self.frames = []
        if not os.path.isfile(timestamps_file):
            raise Exception('time stamps file: {} not found!'.format(timestamps_file))

        images_folder = os.path.join(self.record_folder, 'images')
        if not os.path.isdir(images_folder):
            raise Exception('images folder: {} not found!'.format(images_folder))

        with open(timestamps_file, 'r') as f:
            lines = f.readlines()
            for l in lines:
                if len(l) > 0 and l[0] != '#':
                    sp = l.split()
                    self.frames.append({'time': float(sp[0]),
                                               'image_file': os.path.join(images_folder, sp[1]),
                                               'bbox': None})
        return

    def _get_bbox_ref(self, time_epsilon = 1e-4):
        """
        get bounding box reference
        """

        bbox_ref_file = os.path.join(self.record_folder, 'bbox_reference.txt')
        # if not os.path.isfile(bbox_ref_file):
        #     raise Exception('bbox ref file: {} not found!'.format(bbox_ref_file))

        if os.path.isfile(bbox_ref_file):
            camera_timestamps = np.array([x['time'] for x in self.frames])
            with open(bbox_ref_file, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if len(l) > 0 and l[0] != '#':
                        sp = l.split()
                        timestamp = float(sp[0])
                        bbox = (float(sp[1]), float(sp[2]), float(sp[3]), float(sp[4]))
                        dt = np.abs(camera_timestamps - timestamp)
                        if np.min(dt) <= time_epsilon:
                            idx = np.argmin(dt)
                            self.frames[idx]['bbox'] = bbox

        return

    def draw(self, output_video_file=None):
        """
        draw record frames along with reference bbox if exists

        params:
            output_video_file: video file for saving record video
                               video will not be saved if None
        """

        cv2.namedWindow('recorded frames')

        # set video writer
        if output_video_file is not None:
            print('saving record video to: {}'.format(output_video_file))
            img = cv2.imread(self.frames[0]['image_file'])
            frame_size = img.shape
            frame_times = np.array([x['time'] for x in self.frames])
            dt = frame_times[1:] - frame_times[:-1]
            frame_rate = np.round(1/np.mean(dt))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (int(frame_size[1]), int(frame_size[0])))

        # record all frames
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        for cf in record.frames:
            img = cv2.imread(cf['image_file'])
            img_annotated = img.copy()
            img_annotated = cv2.putText(img_annotated, '{:.3f}'.format(cf['time']), (10, 20), font, fontScale, color=(0, 0, 0), thickness=1)
            if cf['bbox'] is not None:
                pt1 = (int(cf['bbox'][0] - cf['bbox'][2] / 2), int(cf['bbox'][1] - cf['bbox'][3] / 2))
                pt2 = (int(cf['bbox'][0] + cf['bbox'][2] / 2), int(cf['bbox'][1] + cf['bbox'][3] / 2))
                img_annotated = cv2.rectangle(img_annotated, pt1, pt2, color=(200, 200, 50), thickness=1)
            if output_video_file is not None:
                out.write(img)
            cv2.imshow('recorded frames', img_annotated)
            cv2.waitKey(50)

        cv2.destroyAllWindows()
        if output_video_file is not None:
            out.release()
        return



if __name__ == '__main__':

    record_folder = '/home/roee/Projects/datasets/interceptor_drone/common_tests/2025-04-07-08-07-18_gazebo_extracted'

    # analyse and sync record
    record = StandardRecord(record_folder)
    output_video_file = os.path.join(record_folder, 'record_video.avi')
    record.draw(output_video_file=output_video_file)

    print('Done!')
