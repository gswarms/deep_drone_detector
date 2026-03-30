"""Extract information from a rosbag. """

import os
import numpy as np
import cv2
import time
import re
from rosbags.highlevel import AnyReader
# from rosbags.rosbag1 import Writer
# from rosbags.serde import serialize_ros1
# from rosbags.typesys.types import builtin_interfaces__msg__Time as Time
# from rosbags.typesys.types import sensor_msgs__msg__CompressedImage as CompressedImage
# from rosbags.typesys.types import sensor_msgs__msg__Image as Image
# from rosbags.typesys.types import sensor_msgs__msg__Imu as Imu
# from rosbags.typesys.types import std_msgs__msg__Header as Header
import pathlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import common_utils
import roi_utils


class RosBagRecord:
    """
    This object handles LULAV interceptor drone standard bag record
    functionality:
    1) analyses record timing (camera frames and bbox reference)
    2) save rosbag record to standard folder record
    """
    def __init__(self, bag_file, image_topic=None, ref_bbox_topic=None,  detection_polygon_topic=None,
                              detection_results_bbox_topic=None, color_format='RGB'):

        self.frames = None
        self.start_time = None
        self.end_time = None

        self.color_format = color_format

        self.bag_file = bag_file
        if (not os.path.isfile(self.bag_file)) and (not os.path.isdir(self.bag_file)):
            raise Exception('bag file {} not found!'.format(bag_file))
        self.image_topic = image_topic
        self.ref_bbox_topic = ref_bbox_topic

        self.detection_polygon_topic = detection_polygon_topic
        self.detection_results_bbox_topic = detection_results_bbox_topic

        self.frame_times = None
        self.ref_bbox_times = None
        self.detection_input_polygon_data = []
        self.detection_results_data = []
        self.frame_size = None
        return

    def analyse_bag(self):
        """
        analyse bag record
        """

        print(f"\nAnalysing bag file: {self.bag_file}")
        print(f"Reading bag...")

        with AnyReader([pathlib.Path(self.bag_file)]) as self.bag:

            # get frame times
            self.frame_times = None
            self._get_bag_camera_frame_times(self.image_topic)

            # get bbox reference times
            self.ref_bbox_times = None
            self._get_bag_ref_bbox_times(self.ref_bbox_topic)

            self._get_bag_detection_polygon()
            self._get_bag_detection_results()

            # analyse image and bbox record statistics
            print('analysing frame times ...')

            # check camera frames
            print('camera frames:')
            frames_dt = None
            if self.frame_times.size > 0:
                frames_dt = self.frame_times[1:] - self.frame_times[:-1]
                frame_mean_step = np.mean(frames_dt)
                frame_step_std = np.std(frames_dt)
                print('     {} frames: time range: {:.4f}-{:.4f}, time steps: {:.2f}+-{:.3f}  (mean, std)'.format(
                    self.frame_times.size, self.frame_times[0], self.frame_times[-1], frame_mean_step, frame_step_std))

            # check bbox times
            if self.ref_bbox_times.size > 0:
                print('bbox reference:')
                ref_bbox_dt = None
                if self.ref_bbox_times.size > 0:
                    ref_bbox_dt = self.ref_bbox_times[1:] - self.ref_bbox_times[:-1]
                    bbox_mean_step = np.mean(ref_bbox_dt)
                    bbox_step_std = np.std(ref_bbox_dt)
                    print('     {} ref_bbox: time range: {:.4f}-{:.4f}, time steps: {:.2f}+-{:.3f}  (mean, std)'.format(
                        self.ref_bbox_times.size, self.ref_bbox_times[0], self.ref_bbox_times[-1], bbox_mean_step, bbox_step_std))

                # match bbox reference tom images
                print('matching bbox reference to frames:')
                match_count = 0
                for t in self.frame_times:
                    dt = min(abs(t - self.ref_bbox_times))
                    if dt <  frame_mean_step/4:
                        match_count = match_count + 1
                print('     total {} frames, found {} bbox ref matches'.format(len(self.frame_times), match_count))
            else:
                ref_bbox_dt = None

            fig = plt.figure('bag record timing')
            ax1 = plt.Subplot(fig, 211)
            fig.add_subplot(ax1)
            ax2 = plt.Subplot(fig, 212)
            fig.add_subplot(ax2)
            fig.suptitle('time difference between frames', fontsize=15)

            if frames_dt is not None:
                ax1.scatter(range(0, frames_dt.size), frames_dt, color=(0, 0, 1), s=2, alpha=1)
                ax1.set_xlabel(r'frame id', fontsize=12)
                ax1.set_ylabel(r'time difference [sec]', fontsize=12)
                ax1.set_title('camera frame times', fontsize=14)
                ax1.grid(True)

            if ref_bbox_dt is not None:
                ax2.scatter(range(0, ref_bbox_dt.size), ref_bbox_dt, color=(0, 0, 1), s=2, alpha=1)
                ax2.set_xlabel(r'frame id', fontsize=12)
                ax2.set_ylabel(r'time difference [sec]', fontsize=12)
                ax2.set_title('bbox reference times', fontsize=14)
                ax2.grid(True)

            fig.tight_layout()
            plt.pause(0.1)
            plt.show(block=True)

        print('\n')
        return

    def _get_bag_camera_frame_times(self, image_topic):
        """
        get camera frame times and frame id
        check frame ids are unique and sorted
        """

        # get image messages
        frame_times = []
        if image_topic is not None:
            connections = [x for x in self.bag.connections if x.topic == image_topic]
            for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                msg = self.bag.deserialize(rawdata, connection.msgtype)
                camera_time = self.__to_time(msg.header.stamp)
                frame_times.append(camera_time)
        self.frame_times = np.array(frame_times)

        # check if sorted and unique
        if self.frame_times.size > 0:
            is_sorted = np.all(np.sort(self.frame_times) == self.frame_times)
            if not is_sorted:
                raise Exception('left frame ids not sorted!')
            is_unique = self.frame_times.size == np.unique(self.frame_times).size
            if not is_unique:
                raise Exception('left frames not unique!')
        return

    def _get_bag_ref_bbox_times(self, bbox_topic):
        """
        get imu times
        """
        ref_bbox_times = []
        if bbox_topic is not None:
            connections = [x for x in self.bag.connections if x.topic == bbox_topic]
            for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                msg = self.bag.deserialize(rawdata, connection.msgtype)
                ref_bbox_imu_time = self.__to_time(msg.header.stamp)
                ref_bbox_times.append(ref_bbox_imu_time)
        self.ref_bbox_times = np.array(ref_bbox_times)
        return


    def _get_bag_detection_polygon(self):
        """
        get imu times
        """

        polygon_data = []
        if self.detection_polygon_topic is not None:
            connections = [x for x in self.bag.connections if x.topic == self.detection_polygon_topic]
            for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                msg = self.bag.deserialize(rawdata, connection.msgtype)
                if msg.__msgtype__ == 'visualization_msgs/msg/ImageMarker':
                    polygon_time = self.__to_time(msg.header.stamp)
                    polygon_points = np.array([[p.x for p in msg.points], [p.y for p in msg.points]]).T
                    polygon_points = polygon_points[:-1,:]  # last one repeats the first
                    polygon_data.append({'time': polygon_time, 'points': polygon_points})
                else:
                    print('Warning: on topic {} bad msg! expecting {} got {}'.format(self.detection_polygon_topic, 'visualization_msgs/msg/ImageMarker', msg.__msgtype__))
        self.detection_input_polygon_data = polygon_data
        return


    def _get_bag_detection_results(self):
        """
        get imu times
        """
        detection_res_data = []
        if self.detection_polygon_topic is not None:
            connections = [x for x in self.bag.connections if x.topic == self.detection_results_bbox_topic]
            for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                msg = self.bag.deserialize(rawdata, connection.msgtype)
                if msg.__msgtype__ == 'visualization_msgs/msg/ImageMarker':
                    bbox_time = self.__to_time(msg.header.stamp)
                    bbox_points = np.array([[p.x for p in msg.points], [p.y for p in msg.points]]).T
                    bbox_points = bbox_points[:-1,:]  # last one repeats the first
                    detection_res_data.append({'time': bbox_time, 'points': bbox_points})
                else:
                    print('Warning: on topic {} bad msg! expecting {} got {}'.format(self.detection_results_bbox_topic, 'visualization_msgs/msg/ImageMarker', msg.__msgtype__))

        self.detection_results_data = detection_res_data
        return


    def _get_closest_detection_polygon(self, query_times, valid_time_gap):
        """
        get the closest polygon to a specific time

        :param query_times - query times list or scalar
        :param valid_time_gap - valid time gap from query times
                                if the closes entry is not within this time gap, result will be None
        :return: res_polygons - each results is a nX2 np.array of polygon points
                                   list of results if query_times is a list
                                   scalar result if query_times is scalar
        """

        if not isinstance(query_times, list):
            return_scalar = True
            query_times = [query_times]
        else:
            return_scalar = False

        # get detection input polygon
        if len(self.detection_input_polygon_data) > 0:
            polygon_times = np.array([p['time'] for p in self.detection_input_polygon_data])
            # find the closest polygon in time
            res_polygons = []
            for t in query_times:
                idx = np.argmin(np.abs(t - polygon_times))
                if np.abs(t - polygon_times[idx]) < valid_time_gap:
                    res_polygons.append(self.detection_input_polygon_data[idx]['points'])
                else:
                    res_polygons.append(None)
        else:
            res_polygons = [None] * len(query_times)

        if return_scalar:
            res_polygons = res_polygons[0]

        return res_polygons


    def _get_closest_detection_results(self, query_times, valid_time_gap):
        """
        get the closest detection results to a specific time

        :param query_times - query times list or scalar
        :param valid_time_gap - valid time gap from query times
                                if the closes entry is not within this time gap, result will be None
        :return: detection_results - each results is a detection bbox [xtl, ytl, w, h]
                                   list of results if query_times is a list
                                   scalar result if query_times is scalar
        """

        if not isinstance(query_times, list):
            return_scalar = True
            query_times = [query_times]
        else:
            return_scalar = False

        # get detection input polygon
        if len(self.detection_input_polygon_data) > 0:
            detection_result_times = np.array([p['time'] for p in self.detection_results_data])

            # find the closest polygon in time
            detection_results = []
            for t in query_times:
                idx = np.argmin(np.abs(t - detection_result_times))
                if np.abs(t - detection_result_times[idx]) < valid_time_gap:
                    detection_results.append(self.detection_results_data[idx]['points'])
                else:
                    detection_results.append(None)
        else:
            detection_results = [None] * len(query_times)

        if return_scalar:
            detection_results = detection_results[0]

        return detection_results


    @staticmethod
    def __to_time(ros_time_stamp):
        return ros_time_stamp.sec + ros_time_stamp.nanosec * 1e-9

    def save_to_folder(self, output_folder, start_time=None, end_time=None, detection_polygon_output_file=None):
        """
        save camera frames and bbox reference to standard record folder format
        - output folder: output folder
        - start_time, end_time: save only part of the bag record
                                units are in [sec] for start of the record
        """

        print('saving data:')
        print('    output folder: {}'.format(output_folder))

        if start_time is None:
            start_time = -np.inf
        if end_time is None:
            end_time = np.inf
        self.start_time = start_time
        self.end_time = end_time
        print('    taking records from {}[sec] up to {}[sec]'.format(start_time, end_time))

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        with AnyReader([pathlib.Path(self.bag_file)]) as self.bag:

            # save images
            print('   - saving images')
            # self._camera_images_to_folder(output_folder, self.image_topic)
            self._camera_images_to_folder(output_folder, self.image_topic, detection_polygons_output_file=detection_polygon_output_file)

            # save bbox ref data
            self._save_bbox_ref(output_folder, self.ref_bbox_topic)

            if detection_polygon_output_file is not None:
                self._save_detection_input_polygon(detection_polygon_output_file)


        print('Done')
        return


    def save_to_video(self, output_video_file, start_time=None, end_time=None,
                      draw_detection_polygon=False, draw_detection_results=False, draw_frame_id=False):
        """
        save camera frames to video format
        - output_video_file: output video file
        - start_time, end_time: save only part of the bag record
                                units are in [sec] for start of the record
        """

        print('saving images to video:')
        print('    output video file: {}'.format(output_video_file))

        if start_time is None:
            start_time = -np.inf
        if end_time is None:
            end_time = np.inf
        self.start_time = start_time
        self.end_time = end_time
        print('    taking images from {}[sec] up to {}[sec]'.format(start_time, end_time))

        with AnyReader([pathlib.Path(self.bag_file)]) as self.bag:

            # save images
            video_folder = os.path.dirname(output_video_file)
            if not os.path.isdir(video_folder):
                os.makedirs(video_folder)

            video_obj = None

            timestamps_data = []
            frame_id = 0
            if image_topic is not None:
                connections = [x for x in self.bag.connections if x.topic == image_topic]
                for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                    msg = self.bag.deserialize(rawdata, connection.msgtype)
                    frame_time = float(self.__to_time(msg.header.stamp))
                    timestamps_data.append(frame_time)

                    msg_timestamp = timestamp * 1e-9
                    if self.start_time <= msg_timestamp <= self.end_time:
                        # get images
                        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                        if im.shape[2] == 1:
                            cv_img = np.copy(im.squeeze())
                        elif im.shape[2] == 3:
                            if self.color_format.upper() == 'RGB':
                                cv_img = np.copy(im)
                            elif self.color_format.upper() == 'BGR':
                                cv_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            else:
                                raise Exception('invalid color format: {}'.format(self.color_format))
                        else:
                            raise Exception('invalid image!')

                        frames_time_step = np.median(np.diff(self.frame_times))
                        valid_time_gap = frames_time_step * 0.6


                        # get detection input polygon
                        if draw_detection_polygon:
                            polygon_points = self._get_closest_detection_polygon(frame_time, valid_time_gap)
                            if polygon_points is not None:
                                pp = np.round(polygon_points).astype(np.int32)
                                cv_img = cv2.polylines(cv_img, np.array([pp]), isClosed=True,
                                                       color=(255, 0, 0), thickness=1)

                        # get detection results
                        if draw_detection_results:
                            bbox_points = self._get_closest_detection_results(frame_time, valid_time_gap)
                            if bbox_points is not None:
                                bp = np.round(bbox_points).astype(np.int32)
                                cv_img = cv2.polylines(cv_img, np.array([bp]), isClosed=True,
                                                       color=(255, 255, 0), thickness=1)

                        if draw_frame_id:
                            cv2.putText(cv_img, 'frame {}'.format(frame_id), (20, 20),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(100, 255, 255))
                            frame_id = frame_id + 1


                        # write to video
                        if video_obj is None:
                            fourcc = cv2.VideoWriter_fourcc(*"XVID")
                            video_obj = cv2.VideoWriter(output_video_file, fourcc, 20.0, (im.shape[1], im.shape[0]), isColor=True)
                        video_obj.write(cv_img)

                # save timestamps file
                ts = np.array(timestamps_data)
                dt = np.mean(ts[1:] - ts[:-1])
                print('recorded frame rate = {}Hz'.format(1/dt))


        print('Done')
        return

    def _camera_images_to_folder(self, output_folder, image_topic, detection_polygons_output_file=None):
        """
        save camera frames to a folder
        camera_frame_sync: True - use synced times
                           False - use recorded times
        save_only_stereo_frames: True - save only common stereo frames
                                 False - save all frames for each camera
        """

        t_img = 0
        t_poly = 0

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        timestamp_file_basename = os.path.join(output_folder, 'timestamps.txt')

        image_subfolder_name = os.path.join(output_folder, 'images')
        if not os.path.isdir(image_subfolder_name):
            os.makedirs(image_subfolder_name)

        # save images
        max_frame_id = self.frame_times.size
        num_frame_id_digits = int(np.ceil(np.log10(max_frame_id)))

        save_detection_polygons = detection_polygons_output_file is not None
        if save_detection_polygons:
            detection_polygons_output_folder = os.path.dirname(detection_polygons_output_file)
            if not os.path.isdir(detection_polygons_output_folder):
                os.makedirs(detection_polygons_output_folder)
            frames_time_step = np.median(np.diff(self.frame_times))
            valid_time_gap = frames_time_step * 0.6
            self.frame_size = None
            frame_polygons = roi_utils.PolygonPerFrame(frame_size=self.frame_size)

        frame_id_left = 0
        timestamps_data = []
        if image_topic is not None:
            connections = [x for x in self.bag.connections if x.topic == image_topic]
            for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                msg = self.bag.deserialize(rawdata, connection.msgtype)
                frame_time = float(self.__to_time(msg.header.stamp))

                msg_timestamp = timestamp * 1e-9
                if self.start_time <= msg_timestamp <= self.end_time:
                    t1 = time.monotonic()
                    image_file_name = '{:0{x}d}.png'.format(frame_id_left, x=num_frame_id_digits)
                    frame_id_left += 1
                    timestamps_data.append({'time': frame_time, 'image_file': image_file_name})

                    # write images
                    im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                    if im.shape[2] == 1:
                        cv_img = im.squeeze()
                    elif im.shape[2] == 3:
                        if self.color_format.upper() == 'RGB':
                            cv_img = np.copy(im)
                        elif self.color_format.upper() == 'BGR':
                            cv_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        else:
                            raise Exception('invalid color format: {}'.format(self.color_format))
                    else:
                        raise Exception('invalid image!')
                    image_file_path = os.path.join(image_subfolder_name, image_file_name)
                    cv2.imwrite(image_file_path, cv_img)

                    t_img = t_img + (time.monotonic() - t1)

                    if self.frame_size is None:
                        self.frame_size = (im.shape[1], im.shape[0])
                        if save_detection_polygons:
                            frame_polygons.frame_size = self.frame_size
                    elif self.frame_size[0] != im.shape[1] or self.frame_size[1] != im.shape[0]:
                        print('warning: image size ({}x{}) is not consistent with first image size ({}x{})'.format(im.shape[1], im.shape[0], self.frame_size[0], self.frame_size[1]))

                    if save_detection_polygons:
                        t1 = time.monotonic()
                        detection_polygon = self._get_closest_detection_polygon(frame_time, valid_time_gap)
                        if detection_polygon is not None:
                            detection_polygon = detection_polygon.tolist()
                            frame_polygons.set(frame_id_left, detection_polygon)
                        # frame_polygons.save(detection_polygons_output_file)
                        t_poly = t_poly + (time.monotonic() - t1)

            if save_detection_polygons:
                t1 = time.monotonic()
                frame_polygons.save(detection_polygons_output_file)
                t_poly = t_poly + (time.monotonic() - t1)

            print('saving images time = {}sec'.format(t_img))
            print('saving polygons time = {}sec'.format(t_poly))

            # save timestamps file
            camera_left_timestamp_file = os.path.join(image_subfolder_name, timestamp_file_basename)
            self._save_camera_timestamps_to_file(camera_left_timestamp_file, timestamps_data)

        return

    @ staticmethod
    def _save_camera_timestamps_to_file(timestamps_file, timestamps_data):
        """
        save camera frames to a folder
        camera_frame_sync: True - use synced times
                           False - use recorded times
        save_only_stereo_frames: True - save only common stereo frames
                                 False - save all frames for each camera
        """

        output_folder = os.path.dirname(timestamps_file)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        with open(timestamps_file, 'w') as f:
            f.write('#timestamp [ns]    filename\n')
            for ts in timestamps_data:
                f.write('{:.6f} {}\n'.format(float(ts['time']), ts['image_file']))

    def _save_bbox_ref(self, output_folder, ref_bbox_topic):
        """
        save imu data to a text file
        """
        if ref_bbox_topic is not None:
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            output_file = os.path.join(output_folder, 'bbox_reference.txt')

            with open(output_file, 'w') as f:
                if ref_bbox_topic is not None:
                    f.write('# timestamp [ns]	center_x   center_y   size_x   size_y\n')
                    connections = [x for x in self.bag.connections if x.topic == ref_bbox_topic]
                    for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                        msg = self.bag.deserialize(rawdata, connection.msgtype)
                        frame_time = float(self.__to_time(msg.header.stamp))
                        msg_timestamp = timestamp * 1e-9
                        if self.start_time <= msg_timestamp <= self.end_time:
                            if len(msg.detections) > 0:
                                for d in msg.detections:
                                    f.write('{:.6f} {} {} {} {}\n'.format(frame_time,
                                                                            d.bbox.center.position.x, d.bbox.center.position.y,
                                                                            d.bbox.size_x, d.bbox.size_y))

        return


    def _save_detection_input_polygon(self, output_file):
        """
        save detection_input_polygon data to a yaml file
        """
        polygon_data = roi_utils.PolygonPerFrame(self.frame_size)
        for i, p in enumerate(self.detection_input_polygon_data):
            polygon_points = p['points'].tolist()  # xy to pixel, and convert to list
            polygon_time = p['time']
            polygon_data.set(i, polygon_points, polygon_time)
        polygon_data.save(output_file)
        return

    def __del__(self):
        if type(self.bag) == AnyReader and self.bag.isopen:
            self.bag.close()

if __name__ == '__main__':


    # bag_file = '/home/roee/Projects/datasets/interceptor_drone/common_tests/2025-04-07-08-07-18_gazebo/'
    # valid_record_times = {'start': -np.inf, 'end': np.inf}
    # image_topic = '/world/baylands/model/gs001_0/link/base_link/sensor/camera_sensor/image'
    # ref_bbox_topic = '/boxes_2d'
    # output_folder = '/home/roee/Projects/datasets/interceptor_drone/common_tests/2025-04-07-08-07-18_gazebo_extracted/'

    # # bag_file = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_133041/'
    # bag_file = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_140609'
    # valid_record_times = {'start': -np.inf, 'end': np.inf}
    # image_topic = '/camera/image_raw'
    # ref_bbox_topic = None
    # # output_folder = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_133041_extracted/'
    # output_folder = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_140609_extracted/'
    # video_output_file = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250511_140609.avi'

    # bag_file = '/home/roee/Projects/datasets/interceptor_drone/20250519_kfar_galim/2025-05-19_11-38-20_bags/camera_2025_5_19-8_38_27'
    # valid_record_times = {'start': 1747643941, 'end': np.inf}
    # image_topic = '/camera/image_raw'
    # ref_bbox_topic = None
    # output_folder = '/home/roee/Projects/datasets/interceptor_drone/20250519_kfar_galim/camera_20250519_083827_extracted/'
    # video_output_file = '/home/roee/Projects/datasets/interceptor_drone/20250519_kfar_galim/camera_20250519_083827.avi'


    # ------------------ kfar massarik 08.06.2025 ------------------------------

    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_17-30-42/camera_2025_6_8-14_30_56'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-04-53/camera_2025_6_8-15_4_56'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-17-57/camera_2025_6_8-15_18_8'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-29-51/camera_2025_6_8-15_29_54'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-51-15/camera_2025_6_8-15_51_18'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-52-56/camera_2025_6_8-15_52_58'  # ???
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-53-46/camera_2025_6_8-15_53_49'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-58-42/camera_2025_6_8-15_58_44'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-59-46/camera_2025_6_8-15_59_49'  # ???
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-00-25/camera_2025_6_8-16_0_28'  # ???
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-08-34/camera_2025_6_8-16_8_48'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-09-33/camera_2025_6_8-16_9_38'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-17-25/camera_2025_6_8-16_17_28'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-18-31/camera_2025_6_8-16_18_34'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-24-31/camera_2025_6_8-16_24_34'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-25-35/camera_2025_6_8-16_25_38'  # ???

    # ------------------ kfar galim 01.07.2025 ------------------------------
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_08-26-13/camera_2025_7_1-5_26_17'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-01-54/camera_2025_7_1-6_2_3'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-27-36/camera_2025_7_1-6_27_39'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-34-06/camera_2025_7_1-6_34_9'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-35-10/camera_2025_7_1-6_35_13'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-46-51/camera_2025_7_1-6_46_54'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-48-20/camera_2025_7_1-6_48_22'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-03-29/camera_2025_7_1-7_3_32'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-04-48/camera_2025_7_1-7_4_52'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-47-42/camera_2025_7_1-7_47_54'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-49-24/camera_2025_7_1-7_49_27'


    # ------------------ kfar galim 06.07.2025 ------------------------------
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_09-42-30/camera_2025_7_6-6_42_42'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_10-03-00/camera_2025_7_6-7_3_4'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_10-25-45/camera_2025_7_6-7_25_48'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_10-26-27/camera_2025_7_6-7_26_32'  # is this missing?
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_10-33-09/camera_2025_7_6-7_33_16'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_10-33-58/camera_2025_7_6-7_34_2'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_10-34-30/camera_2025_7_6-7_34_36'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_10-49-34/camera_2025_7_6-7_49_37'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_10-50-16/camera_2025_7_6-7_50_20'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_10-53-26/camera_2025_7_6-7_53_29'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_10-54-07/camera_2025_7_6-7_54_30'  # bug with roi polygon!
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_11-07-57/camera_2025_7_6-8_8_2'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_11-08-38/camera_2025_7_6-8_8_42'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_11-14-42/camera_2025_7_6-8_14_49'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_11-15-43/camera_2025_7_6-8_15_46'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_11-20-06/camera_2025_7_6-8_20_9'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250706_kfar_galim/2025-07-06_11-20-52/camera_2025_7_6-8_20_56'  # is this missing?


    # ------------------ kfar galim 10.07.2025 ------------------------------
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_07-40-11/camera_2025_7_10-4_40_14'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_07-54-07/camera_2025_7_10-4_54_11'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_08-01-21/camera_2025_7_10-5_1_32'  # 2X0.1
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_08-12-42/camera_2025_7_10-5_12_45'  # 1X0.1
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_08-24-03/camera_2025_7_10-5_24_17'  # 1X0.1
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_08-32-49/camera_2025_7_10-5_32_52'  # 4X0.1
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_09-17-03/camera_2025_7_10-6_17_15'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_09-17-03/camera_2025_7_10-6_17_15'  # bad
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250710_kfar_galim/2025-07-10_09-17-49/camera_2025_7_10-6_17_53'  # bad

    # ------------------ kfar galim 16.07.2025 ------------------------------
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_08-42-21/camera_2025_7_16-5_42_34'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_09-16-28/camera_2025_7_16-6_16_31'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_09-17-35/camera_2025_7_16-6_17_39'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_09-18-14/camera_2025_7_16-6_18_18'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_09-18-58/camera_2025_7_16-6_19_1'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_11-03-44/camera_2025_7_16-8_3_56'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_11-04-35/camera_2025_7_16-8_4_38'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_11-05-19/camera_2025_7_16-8_5_23'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250716_kfar_galim/2025-07-16_11-05-41/camera_2025_7_16-8_5_53'  # bad bag

    # ------------------ kfar galim 30.07.2025 ------------------------------
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_07-59-08/camera_2025_7_30-4_59_23'  # 6 frame missed 0.1sec
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-13-53/camera_2025_7_30-5_13_58'  # 4 frame missed 0.1sec
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-24-20/camera_2025_7_30-5_24_24'  # 3 frame missed 0.1sec
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-25-35/camera_2025_7_30-5_25_39'  # 2 frame missed 0.1sec
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-42-38/camera_2025_7_30-5_42_41'  # no frame misses
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-48-07/camera_2025_7_30-5_48_17'  # 7 frame missed 0.1sec
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_08-50-03/camera_2025_7_30-5_50_7'  # 1 frame missed 0.1sec
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20250730_kfar_galim/2025-07-30_09-45-04/camera_2025_7_30-6_45_8'  # bad bag

    # ------------------ kfar galim 30.07.2025 ------------------------------
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20251005_kfar_galim/20251005_161114/camera_20251005_1611'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20251005_kfar_galim/20251005_161216/camera_20251005_1612'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20251005_kfar_galim/20251005_161309/camera_20251005_1613'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20251005_kfar_galim/20251005_161344/camera_20251005_1613'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20251005_kfar_galim/20251005_163355/camera_20251005_1633'
    # bag_folder = '/home/roee/Projects/datasets/interceptor_drone/20251005_kfar_galim/20251005_163529/camera_20251005_1635'

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5 common dataset start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # ------------------ 20251214_reshafim ------------------------------  *** clr_format = 'BGR'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251214_reshafim'
    # bag_folder = os.path.join(base_folder, '20251214_1124_19/20251214_1234_56/camera_20251214_1335')  # bad
    # bag_folder = os.path.join(base_folder, '20251214_1232_41/20251214_1233_08/camera_20251214_1333')
    # bag_folder = os.path.join(base_folder, '20251214_1232_41/20251214_1234_56/camera_20251214_1335')  # bad
    # bag_folder = os.path.join(base_folder, '20251214_1232_41/20251214_1235_55/camera_20251214_1336')
    # bag_folder = os.path.join(base_folder, '20251214_1232_41/20251214_1236_52/camera_20251214_1336')  # bad
    # bag_folder = os.path.join(base_folder, '20251214_1322_23/20251214_1325_35/camera_20251214_1425')  # bad
    # bag_folder = os.path.join(base_folder, '20251214_1322_23/20251214_1327_07/camera_20251214_1427')


    # ------------------ 20251208_reshafim ------------------------------  *** clr_format = 'BGR'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251208_reshafim'
    # bag_folder = os.path.join(base_folder, '20251208_1155_58/20251208_1157_42/camera_20251208_1257')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1155_58/20251208_1200_23/camera_20251208_1300')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1155_58/20251208_1204_12/camera_20251208_1304')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1155_58/20251208_1206_00/camera_20251208_1306')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1155_58/20251208_1207_52/camera_20251208_1307')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1216_03/20251208_1216_29/camera_20251208_1316')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1235_31/20251208_1239_35/camera_20251208_1339')
    # bag_folder = os.path.join(base_folder, '20251208_1246_16/20251208_1247_59/camera_20251208_1348')
    # bag_folder = os.path.join(base_folder, '20251208_1256_17/20251208_1257_05/camera_20251208_1357')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1256_17/20251208_1259_11/camera_20251208_1359')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1256_17/20251208_1301_26/camera_20251208_1401')
    # bag_folder = os.path.join(base_folder, '20251208_1256_17/20251208_1303_25/camera_20251208_1403')
    # bag_folder = os.path.join(base_folder, '20251208_1327_30/20251208_1332_56/camera_20251208_1433')
    # bag_folder = os.path.join(base_folder, '20251208_1454_32/20251208_1457_05/camera_20251208_1557')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1454_32/20251208_1458_20/camera_20251208_1558')
    # bag_folder = os.path.join(base_folder, '20251208_1454_32/20251208_1500_28/camera_20251208_1600')
    # bag_folder = os.path.join(base_folder, '20251208_1454_32/20251208_1501_47/camera_20251208_1601')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1509_37/20251208_1511_41/camera_20251208_1611')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1509_37/20251208_1513_17/camera_20251208_1613')
    # bag_folder = os.path.join(base_folder, '20251208_1509_37/20251208_1514_28/camera_20251208_1614')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1509_37/20251208_1515_50/camera_20251208_1615')
    # bag_folder = os.path.join(base_folder, '20251208_1509_37/20251208_1516_53/camera_20251208_1616')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1521_09/20251208_1523_58/camera_20251208_1624')
    # bag_folder = os.path.join(base_folder, '20251208_1521_09/20251208_1525_11/camera_20251208_1625')  # bad
    # bag_folder = os.path.join(base_folder, '20251208_1521_09/20251208_1526_27/camera_20251208_1626')


    # ------------------ kfar_galim 04.12.2025 ------------------------------  *** clr_format = 'BGR'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251204_kfar_galim'
    # bag_folder = os.path.join(base_folder, '20251204_1423_16/20251204_1423_36/camera_20251204_1523')  # bad
    # bag_folder = os.path.join(base_folder, '20251204_1423_16/20251204_1425_28/camera_20251204_1525')  # bad
    # bag_folder = os.path.join(base_folder, '20251204_1423_16/20251204_1431_50/camera_20251204_1531')  # bad
    # bag_folder = os.path.join(base_folder, '20251204_1438_06/20251204_1438_28/camera_20251204_1538')  # bad
    # bag_folder = os.path.join(base_folder, '20251204_1453_40/20251204_1454_07/camera_20251204_1554')
    # bag_folder = os.path.join(base_folder, '20251204_1511_13/20251204_1511_34/camera_20251204_1611')
    # bag_folder = os.path.join(base_folder, '20251204_1511_13/20251204_1515_28/camera_20251204_1615')
    # bag_folder = os.path.join(base_folder, '20251204_1526_11/20251204_1527_23/camera_20251204_1627')
    # bag_folder = os.path.join(base_folder, '20251204_1526_11/20251204_1532_30/camera_20251204_1632')
    # bag_folder = os.path.join(base_folder, '20251204_1550_34/20251204_1550_54/camera_20251204_1650')
    # bag_folder = os.path.join(base_folder, '20251204_1600_55/20251204_1601_20/camera_20251204_1701')
    # bag_folder = os.path.join(base_folder, '20251204_1600_55/20251204_1603_11/camera_20251204_1703')


    # ------------------ kfar_galim 26.11.2025 ------------------------------  *** clr_format = 'RGB'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251126_kfar_galim'
    # bag_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1509_36/camera_20251126_1609')  # bad
    # bag_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1510_20/camera_20251126_1610')
    # bag_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1511_16/camera_20251126_1611')
    # bag_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1512_12/camera_20251126_1612')  # bad
    # bag_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1513_05/camera_20251126_1613')
    # bag_folder = os.path.join(base_folder, '20251126_1507_31/20251126_1514_12/camera_20251126_1614')  # bad
    # bag_folder = os.path.join(base_folder, '20251126_1523_01/20251126_1524_04/camera_20251126_1624')  # bad
    # bag_folder = os.path.join(base_folder, '20251126_1523_01/20251126_1526_12/camera_20251126_1626')
    # bag_folder = os.path.join(base_folder, '20251126_1523_01/20251126_1527_11/camera_20251126_1627')  # bad
    # bag_folder = os.path.join(base_folder, '20251126_1536_04/20251126_1536_50/camera_20251126_1636')
    # bag_folder = os.path.join(base_folder, '20251126_1558_12/20251126_1558_24/camera_20251126_1658')  # bad
    # bag_folder = os.path.join(base_folder, '20251126_1558_12/20251126_1600_23/camera_20251126_1700')
    # bag_folder = os.path.join(base_folder, '20251126_1603_18/20251126_1603_41/camera_20251126_1703')  # bad
    # bag_folder = os.path.join(base_folder, '20251126_1603_18/20251126_1604_48/camera_20251126_1704')
    # bag_folder = os.path.join(base_folder, '20251126_1603_18/20251126_1605_59/camera_20251126_1706')  # bad
    # bag_folder = os.path.join(base_folder, '20251126_1603_18/20251126_1607_28/camera_20251126_1707')  # bad
    # bag_folder = os.path.join(base_folder, '20251126_1603_18/20251126_1608_34/camera_20251126_1708')

    # ------------------ kfar_galim 27.10.2025 ------------------------------  *** clr_format = 'RGB'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20251027_kfar_galim'
    # bag_folder = os.path.join(base_folder, '20251027_123000/camera_20251027_1230')


    # ------------------ lehavim 18.09.2025 ------------------------------  *** clr_format = 'RGB'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250918_lehavim'

    # bag_folder = os.path.join(base_folder, 'hb002/20250910_1441_06/camera_20250910_1441')  # bad
    # bag_folder = os.path.join(base_folder, 'hb002/20250910_1442_03/camera_20250910_1442')  # bad
    # bag_folder = os.path.join(base_folder, 'hb002/20250915_1019_36/camera_20250915_1019')
    # bag_folder = os.path.join(base_folder, 'hb002/20250918_0941_14/camera_20250918_0941')  # bad
    # bag_folder = os.path.join(base_folder, 'hb002/20250918_1005_37/camera_20250918_1005')  # bad
    # bag_folder = os.path.join(base_folder, 'hb002/20250918_1008_19/camera_20250918_1008')  # bad
    # bag_folder = os.path.join(base_folder, 'hb002/20250918_1010_22/camera_20250918_1010')  # bad
    # bag_folder = os.path.join(base_folder, 'hb002/20250918_1040_02/camera_20250918_1040')
    # bag_folder = os.path.join(base_folder, 'hb002/20250918_1041_26/camera_20250918_1041')
    # bag_folder = os.path.join(base_folder, 'hb002/20250918_1043_09/camera_20250918_1043')  # bad
    # bag_folder = os.path.join(base_folder, 'hb002/20250918_1044_49/camera_20250918_1044')
    # bag_folder = os.path.join(base_folder, 'hb002/20250918_1046_44/camera_20250918_1046')  # bad

    # bag_folder = os.path.join(base_folder, 'hb004/20250918_1439_53/camera_20250918_1440')  # bad
    # bag_folder = os.path.join(base_folder, 'hb004/20250918_1441_41/camera_20250918_1441')  # bad
    # bag_folder = os.path.join(base_folder, 'hb004/20250918_1443_47/camera_20250918_1443')  # bad
    # bag_folder = os.path.join(base_folder, 'hb004/20250918_1446_11/camera_20250918_1446')  # bad
    # bag_folder = os.path.join(base_folder, 'hb004/20250918_1451_40/camera_20250918_1451')
    # bag_folder = os.path.join(base_folder, 'hb004/20250918_1453_47/camera_20250918_1453')
    # bag_folder = os.path.join(base_folder, 'hb004/20250918_1456_00/camera_20250918_1456')
    # bag_folder = os.path.join(base_folder, 'hb004/20250918_1457_57/camera_20250918_1458')  # bad

    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1154_08/camera_20250918_1154')
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1156_18/camera_20250918_1156')
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1158_27/camera_20250918_1158')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1200_17/camera_20250918_1200')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1210_46/camera_20250918_1210')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1216_55/camera_20250918_1217')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1219_00/camera_20250918_1219')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1232_08/camera_20250918_1232')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1235_26/camera_20250918_1235')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1239_54/camera_20250918_1239')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1248_01/camera_20250918_1248')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1249_56/camera_20250918_1249')
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1252_07/camera_20250918_1252')
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1256_02/camera_20250918_1256')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1336_04/camera_20250918_1336')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1338_16/camera_20250918_1338')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1341_50/camera_20250918_1341')  # bad
    # bag_folder = os.path.join(base_folder, 'pz004/20250918_1410_23/camera_20250918_1411')  # bad

    # ------------------ lehavim 17.09.2025 ------------------------------  *** clr_format = 'RGB'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250917_lehavim'
    # bag_folder = os.path.join(base_folder, 'hb003/20250915_1028_24/camera_20250915_1028')
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1506_17/camera_20250917_1506')  # bad
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1547_12/camera_20250917_1547')  # bad
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1549_23/camera_20250917_1549')  # bad
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1557_10/camera_20250917_1557')  # bad
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1623_52/camera_20250917_1624')  # bad
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1625_13/camera_20250917_1625')  # bad
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1645_47/camera_20250917_1645')  # bad
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1657_35/camera_20250917_1657')  # bad
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1715_18/camera_20250917_1715')  # bad
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1747_34/camera_20250917_1747')
    # bag_folder = os.path.join(base_folder, 'hb003/20250917_1754_26/camera_20250917_1754')  # bad

    # ------------------ kfar_masarik 08.06.2025 ------------------------------  *** clr_format = 'RGB'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250608_kfar_masarik'
    # bag_folder = os.path.join(base_folder, '2025-06-08_17-30-42/camera_2025_6_8-14_30_56')  # bad
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-04-53/camera_2025_6_8-15_4_56')
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-17-57/camera_2025_6_8-14_30_56')  # bad
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-17-57/camera_2025_6_8-15_18_8')  # bad
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-29-51/camera_2025_6_8-15_29_54')
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-41-19/camera_2025_6_8-15_41_22')  # bad
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-45-39/camera_2025_6_8-15_45_44')  # bad
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-51-15/camera_2025_6_8-15_51_18')
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-52-56/camera_2025_6_8-15_52_58')
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-53-46/camera_2025_6_8-15_53_49')
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-58-42/camera_2025_6_8-15_58_44')  # bad
    # bag_folder = os.path.join(base_folder, '2025-06-08_18-59-46/camera_2025_6_8-15_59_49')
    # bag_folder = os.path.join(base_folder, '2025-06-08_19-00-25/camera_2025_6_8-16_0_28')
    # bag_folder = os.path.join(base_folder, '2025-06-08_19-08-34/camera_2025_6_8-16_8_48')
    # bag_folder = os.path.join(base_folder, '2025-06-08_19-09-33/camera_2025_6_8-16_9_38')
    # bag_folder = os.path.join(base_folder, '2025-06-08_19-17-25/camera_2025_6_8-16_17_28')  # bad
    # bag_folder = os.path.join(base_folder, '2025-06-08_19-18-31/camera_2025_6_8-16_18_34')  # bad
    # bag_folder = os.path.join(base_folder, '2025-06-08_19-24-31/camera_2025_6_8-16_24_34')  # bad
    # bag_folder = os.path.join(base_folder, '2025-06-08_19-25-35/camera_2025_6_8-16_25_38')

    # ------------------ hadera 21.04.2025 ------------------------------  *** clr_format = 'RGB'
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20251211/20250421_hadera'
    # bag_folder = os.path.join(base_folder, '2025-04-21_10-57-38/camera_2025_4_21-7_59_8')
    # bag_folder = os.path.join(base_folder, '2025-04-21_10-59-30/camera_2025_4_21-7_59_41')
    # bag_folder = os.path.join(base_folder, '2025-04-21_11-11-25/camera_2025_4_21-8_12_28')
    # bag_folder = os.path.join(base_folder, '2025-04-21_11-13-03/camera_2025_4_21-8_13_40')
    # bag_folder = os.path.join(base_folder, '2025-04-21_11-24-25/camera_2025_4_21-8_25_32')
    # bag_folder = os.path.join(base_folder, '2025-04-21_11-26-29/camera_2025_4_21-8_26_42')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5 common dataset end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    # ------------------ reshafim 01.03.2026 ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260103_reshafim'
    # # bag_folder = os.path.join(base_folder, '20260103_1022_41/camera_20260103_1122')
    # bag_folder = os.path.join(base_folder, '20260103_1024_30/camera_20260103_1124')


    # ------------------ reshafim 20.01.2026 ------------------------------ (rc-plane)
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260120_reshafim'
    # bag_folder = os.path.join(base_folder, '20260120_1005_54/20260120_1006_12/camera_20260120_1106')
    # bag_folder = os.path.join(base_folder, '20260120_1029_46/20260120_1032_24/camera_20260120_1132')
    # bag_folder = os.path.join(base_folder, '20260120_1029_46/20260120_103322/camera_20260120_1133')
    # bag_folder = os.path.join(base_folder, '20260120_1110_10/20260120_1111_10/camera_20260120_1211')
    # bag_folder = os.path.join(base_folder, '20260120_1157_26/20260120_1158_03/camera_20260120_1258')  # bad magic
    # bag_folder = os.path.join(base_folder, '20260120_1258_00/camera_20260120_1258')  # bad magic


    # ------------------ 309 25.02.2026 ------------------------------ (rc-plane)
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260225_309'
    # bag_folder = os.path.join(base_folder, '20260225_0911_49/20260225_0912_27/camera_20260225_1012')
    # bag_folder = os.path.join(base_folder, '20260225_0911_49/20260225_0914_09/camera_20260225_1014')  # bad bag magic
    # bag_folder = os.path.join(base_folder, '20260225_0911_49/20260225_0915_22/camera_20260225_1015')
    # bag_folder = os.path.join(base_folder, '20260225_0911_49/20260225_0914_09/camera_20260225_1014')  # bad bag magic
    # bag_folder = os.path.join(base_folder, '20260225_0911_49/20260225_0915_22/camera_20260225_1015')
    # bag_folder = os.path.join(base_folder, '20260225_1233_44/20260225_1234_37/camera_20260225_1334')  # bad bag magic
    # bag_folder = os.path.join(base_folder, '20260225_1233_44/20260225_1235_57/camera_20260225_1336')  # bad bag magic
    # bag_folder = os.path.join(base_folder, '20260225_1233_44/20260225_1237_23/camera_20260225_1337')
    # bag_folder = os.path.join(base_folder, '20260225_1347_29/20260225_1348_54/camera_20260225_1448')
    # bag_folder = os.path.join(base_folder, '20260225_1347_29/20260225_1351_30/camera_20260225_1451')  # bad bag magic

    # ------------------ 20260322_reshafim ------------------------------ (10" quadrotor target)
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260322_reshafim/'
    # bag_folder = os.path.join(base_folder, '20260322_1459_23/camera_20260322_1559')
    # valid_record_times = {'start': 1774184384.332746792, 'end': 1774184393.383570394}
    # valid_record_times = {'start': 1774184396.711760611, 'end': 1774184405.281982002}
    # valid_record_times = {'start': 1774184408.932056670, 'end': 1774184418.157677743 }

    # ------------------ 20260218_reshafim ------------------------------ (18" quadrotor target)
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260218_reshafim'
    # bag_folder = os.path.join(base_folder, '20260218_1019_25/20260218_1020_18/camera_20260218_1120')
    # bag_folder = os.path.join(base_folder, '20260218_1042_35/20260218_1044_43/camera_20260218_1144')  # bad magic
    # bag_folder = os.path.join(base_folder, '20260218_1123_37/20260218_1124_44/camera_20260218_1225')
    # bag_folder = os.path.join(base_folder, '20260218_1306_42/20260218_1307_22/camera_20260218_1407')
    # bag_folder = os.path.join(base_folder, '20260218_1306_42/20260218_1308_29/camera_20260218_1408')  # bad magic
    # bag_folder = os.path.join(base_folder, '20260218_1306_42/20260218_1309_26/camera_20260218_1409')

    # ------------------ 20260215_reshafim ------------------------------ (rc plane)
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260215_reshafim'
    # bag_folder = os.path.join(base_folder, '20260215_1306_22/20260215_1310_35/camera_20260215_1410')  # bad magic
    # bag_folder = os.path.join(base_folder, '20260215_1424_51/20260215_1425_49/camera_20260215_1526')  # bad magic
    # bag_folder = os.path.join(base_folder, '20260215_1424_51/20260215_1428_31/camera_20260215_1528')
    # bag_folder = os.path.join(base_folder, '20260215_1424_51/20260215_1431_14/camera_20260215_1531')  # bad magic
    # bag_folder = os.path.join(base_folder, '20260215_1510_13/20260215_1511_47/camera_20260215_1611')  # bad magic
    # bag_folder = os.path.join(base_folder, '20260215_1537_35/20260215_1537_53/camera_20260215_1637')  # bad magic
    # bag_folder = os.path.join(base_folder, '20260215_1537_35/20260215_1539_19/camera_20260215_1639')  # bad magic

    # ------------------ 20260209_reshafim ------------------------------ (rc plane / 18" quadrotor - lense out of focus)
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim'
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/20260209_0920_32/20260209_0923_57/camera_20260209_1024')
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/20260209_0920_32/20260209_0924_58/camera_20260209_1025')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/20260209_1039_12/20260209_1043_28/camera_20260209_1143')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/20260209_1337_18/20260209_1339_17/camera_20260209_1439')
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/20260209_1337_18/20260209_1341_32/camera_20260209_1441')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/20260209_1620_51/20260209_1623_40/camera_20260209_1723')
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1122_55/camera_20260209_1223')
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1134_23/camera_20260209_1234')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1134_57/camera_20260209_1235')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1137_07/camera_20260209_1237')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1138_29/camera_20260209_1238')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1316_17/camera_20260209_1416')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1339_17/camera_20260209_1439')
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1341_32/camera_20260209_1441')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1621_53/camera_20260209_1722')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1622_32/camera_20260209_1722')  # bad magic
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1623_40/camera_20260209_1723')
    # bag_folder = os.path.join(base_folder, '/home/roee/Projects/datasets/interceptor_drone/20260209_reshafim/hb005/20260209_1625_12/camera_20260209_1725')  # bad magic


    # ------------------ 20260208_reshafim ------------------------------ (rc plane / 18" quadrotor - lense out of focus)
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260208_reshafim'
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1346_25/camera_20260208_1446')  # bad magic
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1348_54/camera_20260208_1449')  # bad magic
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1351_03/camera_20260208_1451')
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1419_19/camera_20260208_1519')
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1513_51/camera_20260208_1614')  # bad magic
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1517_47/camera_20260208_1617')
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1531_15/camera_20260208_1631')  # bad magic
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1558_39/camera_20260208_1658')  # bad magic
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1600_22/camera_20260208_1700')
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1602_30/camera_20260208_1702')
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1616_00/camera_20260208_1716')  # bad magic
    # bag_folder = os.path.join(base_folder, 'hb013/20260208_1620_24/camera_20260208_1720')


    # ------------------ 20260324_reshafim ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260324_reshafim'
    # bag_folder = os.path.join(base_folder, '20260324_1029_26/20260324_1031_04/camera_20260324_1131')
    # valid_record_times = {'start': 1774341088.511273944, 'end': 1774341107.831755309}
    # valid_record_times = {'start': 1774341108.822549225, 'end': 1774341140.032557584}
    # valid_record_times = {'start': 1774341143.203098115, 'end': 1774341151.129449444}
    # valid_record_times = {'start': 1774341162.622658872, 'end': 1774341175.800217957}
    # bag_folder = os.path.join(base_folder, '20260324_1037_17/20260324_1037_40/camera_20260324_1137')
    # valid_record_times = {'start': 1774341491.030302707, 'end': 1774341507.485129842}
    # valid_record_times = {'start': 1774341553.300530886, 'end': 1774341561.366622620}
    # valid_record_times = {'start': 1774341573.196890495, 'end': 1774341595.727706104}
    # bag_folder = os.path.join(base_folder, '20260324_1037_17/20260324_1042_30/camera_20260324_1142')
    # valid_record_times = {'start': 1774341774.101288827, 'end': 1774341793.229704096}
    # valid_record_times = {'start': 1774341809.108145897, 'end': 1774341815.515236448}
    # valid_record_times = {'start': 1774341850.986375442, 'end': 1774341863.614843775}
    # bag_folder = os.path.join(base_folder, '20260324_1048_29/20260324_1049_06/camera_20260324_1149')
    # valid_record_times = {'start': 1774342179.798100476, 'end': 1774342200.653628793}
    # valid_record_times = {'start': 1774342209.228744498, 'end': 1774342223.838200883}

    # ------------------ 20260325_reshafim ------------------------------
    # base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260325_reshafim'
    # bag_folder = os.path.join(base_folder, '20260325_1206_34/20260325_1207_03/camera_20260325_1307')
    # valid_record_times = {'start': 1774433258.376069064, 'end': 1774433273.599835676}
    # valid_record_times = {'start': 1774433288.305787097, 'end': 1774433295.348073693}
    # valid_record_times = {'start': 1774433308.604142580, 'end': 1774433315.439303099}
    # valid_record_times = {'start': 1774433327.038363375, 'end': 1774433334.391339085}
    # valid_record_times = {'start': 1774433341.330062643, 'end': 1774433350.547173041}

    # ------------------ 20260329_reshafim ------------------------------
    base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260329_reshafim'
    # bag_folder = os.path.join(base_folder, '20260329_1404_43/20260329_1405_07/camera_20260329_1405')
    # bag_folder = os.path.join(base_folder, '20260329_1404_43/20260329_1407_02/camera_20260329_1407')
    # bag_folder = os.path.join(base_folder, '20260329_1404_43/20260329_1405_07/camera_20260329_1405')
    # bag_folder = os.path.join(base_folder, '20260329_1411_41/20260329_1411_56/camera_20260329_1412')
    # bag_folder = os.path.join(base_folder, '20260329_1445_19/20260329_1446_24/camera_20260329_1446')
    # bag_folder = os.path.join(base_folder, '20260329_1445_19/20260329_1448_12/camera_20260329_1448')
    bag_folder = os.path.join(base_folder, '20260329_1516_47/20260329_1518_30/camera_20260329_1518')

    valid_record_times = {'start': -np.inf, 'end': np.inf}
    image_topic = '/camera/image_raw'
    ref_bbox_topic = None
    detection_polygon_topic='/detection/visualization/roi_bounding_box'
    detection_results_bbox_topic='/detection/visualization/target_bounding_box'

    scen_folder = os.path.dirname(bag_folder)
    output_folder = bag_folder + '_extracted'
    video_output_file = bag_folder + '.avi'
    # scen_name = os.path.basename(os.path.dirname(bag_folder))
    scen_name = common_utils.path_to_scenario_name(os.path.join(bag_folder,'..'))
    detection_polygon_output_file = os.path.join(output_folder, scen_name + '_recorded_detection_roi_polygons.yaml')
    frame_size = (640, 480)
    clr_format = 'RGB'  # 'BGR' / 'RGB'



    # analyse and sync record
    ros_record = RosBagRecord(bag_folder, image_topic=image_topic, ref_bbox_topic=ref_bbox_topic,
                              detection_polygon_topic=detection_polygon_topic,
                              detection_results_bbox_topic=detection_results_bbox_topic,
                              color_format=clr_format)
    ros_record.analyse_bag()

    t1 = time.monotonic()
    ros_record.save_to_folder(output_folder, start_time=valid_record_times['start'], end_time=valid_record_times['end'],
                              detection_polygon_output_file = detection_polygon_output_file)
    # ros_record.save_to_folder(output_folder, start_time=valid_record_times['start'], end_time=valid_record_times['end'],
    #                           detection_polygon_output_file = None)

    print('save_to_folder time - {}sec'.format(time.monotonic()-t1))

    t1 = time.monotonic()
    ros_record.save_to_video(video_output_file, start_time=valid_record_times['start'], end_time=valid_record_times['end'],
                             draw_detection_polygon=True, draw_detection_results=True, draw_frame_id=True)
    print('save_to_video time - {}sec'.format(time.monotonic()-t1))

    print('Done!')
