"""Extract information from a rosbag. """

import os
import numpy as np
import cv2
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


class RosBagRecord:
    """
    This object handles LULAV interceptor drone standard bag record
    functionality:
    1) analyses record timing (camera frames and bbox reference)
    2) save rosbag record to standard folder record
    """
    def __init__(self, bag_file, image_topic=None, ref_bbox_topic=None):

        self.frames = None
        self.start_time = None
        self.end_time = None

        self.bag_file = bag_file
        if (not os.path.isfile(self.bag_file)) and (not os.path.isdir(self.bag_file)):
            raise Exception('bag file {} not found!'.format(bag_file))
        self.image_topic = image_topic
        self.ref_bbox_topic = ref_bbox_topic

        self.frame_times = None
        self.ref_bbox_times = None
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

    @staticmethod
    def __to_time(ros_time_stamp):
        return ros_time_stamp.sec + ros_time_stamp.nanosec * 1e-9

    def save_to_folder(self, output_folder, start_time=None, end_time=None):
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
            self._camera_images_to_folder(output_folder, self.image_topic)

            # save bbox ref data
            self._save_bbox_ref(output_folder, self.ref_bbox_topic)

        print('Done')
        return


    def save_to_video(self, output_video_file, start_time=None, end_time=None):
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
            if image_topic is not None:
                connections = [x for x in self.bag.connections if x.topic == image_topic]
                for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                    msg = self.bag.deserialize(rawdata, connection.msgtype)
                    frame_time = float(self.__to_time(msg.header.stamp))
                    timestamps_data.append(frame_time)

                    msg_timestamp = timestamp * 1e-9
                    if self.start_time <= msg_timestamp <= self.end_time:
                        # write images
                        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                        if im.shape[2] == 1:
                            cv_img = im.squeeze()
                        elif im.shape[2] == 3:
                            pass
                            cv_img = im
                            # cv_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                            # raise Exception('got color image! expecting grayscale!')
                        else:
                            raise Exception('invalid image!')

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

    def _camera_images_to_folder(self, output_folder, image_topic):
        """
        save camera frames to a folder
        camera_frame_sync: True - use synced times
                           False - use recorded times
        save_only_stereo_frames: True - save only common stereo frames
                                 False - save all frames for each camera
        """

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        timestamp_file_basename = os.path.join(output_folder, 'timestamps.txt')

        image_subfolder_name = os.path.join(output_folder, 'images')
        if not os.path.isdir(image_subfolder_name):
            os.makedirs(image_subfolder_name)

        # save images
        max_frame_id = self.frame_times.size
        num_frame_id_digits = int(np.ceil(np.log10(max_frame_id)))

        frame_id_left = 0
        timestamps_data = []
        if image_topic is not None:
            connections = [x for x in self.bag.connections if x.topic == image_topic]
            for connection, timestamp, rawdata in self.bag.messages(connections=connections):
                msg = self.bag.deserialize(rawdata, connection.msgtype)
                frame_time = float(self.__to_time(msg.header.stamp))

                msg_timestamp = timestamp * 1e-9
                if self.start_time <= msg_timestamp <= self.end_time:
                    image_file_name = '{:0{x}d}.png'.format(frame_id_left, x=num_frame_id_digits)
                    frame_id_left += 1
                    timestamps_data.append({'time': frame_time, 'image_file': image_file_name})

                    # write images
                    im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                    if im.shape[2] == 1:
                        cv_img = im.squeeze()
                    elif im.shape[2] == 3:
                        pass
                        cv_img = im
                        # cv_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        # raise Exception('got color image! expecting grayscale!')
                    else:
                        raise Exception('invalid image!')
                    image_file_path = os.path.join(image_subfolder_name, image_file_name)
                    cv2.imwrite(image_file_path, cv_img)

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

    bag_file = '/home/roee/Projects/datasets/interceptor_drone/20250519_kfar_galim/2025-05-19_11-38-20_bags/camera_2025_5_19-8_38_27'
    valid_record_times = {'start': 1747643941, 'end': np.inf}
    image_topic = '/camera/image_raw'
    ref_bbox_topic = None
    output_folder = '/home/roee/Projects/datasets/interceptor_drone/20250519_kfar_galim/camera_20250519_083827_extracted/'
    video_output_file = '/home/roee/Projects/datasets/interceptor_drone/20250519_kfar_galim/camera_20250519_083827.avi'


    # analyse and sync record
    ros_record = RosBagRecord(bag_file, image_topic=image_topic, ref_bbox_topic=ref_bbox_topic)
    ros_record.analyse_bag()
    ros_record.save_to_folder(output_folder, start_time=valid_record_times['start'], end_time=valid_record_times['end'])
    ros_record.save_to_video(video_output_file, start_time=valid_record_times['start'], end_time=valid_record_times['end'])

    print('Done!')
