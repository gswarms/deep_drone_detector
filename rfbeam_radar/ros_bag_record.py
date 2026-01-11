"""Extract information from a rosbag. """

import os
import numpy as np
import cv2
import time
import re
from collections import Counter
from rosbags.highlevel import AnyReader
import pathlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


class RosBagRecord:
    """
    This object handles LULAV interceptor drone kmd7 radar data bag record
    functionality:
    1) analyses record timing
    2) save record to yaml file
    """
    def __init__(self, bag_file, kmd7_raw_topic=None, kmd7_track_topic=None):
        self.bag_file = bag_file
        if (not os.path.isfile(self.bag_file)) and (not os.path.isdir(self.bag_file)):
            raise Exception('bag file {} not found!'.format(bag_file))
        self.kmd7_raw_topic = kmd7_raw_topic
        self.kmd7_raw_msg_type = 'kmd7/msg/RawDetections'
        self.kmd7_track_topic = kmd7_track_topic
        self.kmd7_track_msg_type = 'kmd7/msg/TrackedDetections'

        self.raw_record_times = []
        self.raw_detections = []
        self.track_record_times = []
        self.track_detections = []

        with AnyReader([pathlib.Path(self.bag_file)]) as self.bag:
            self._get_kmd7_raw()
            self._get_kmd7_track()

        return


    def analyse_bag(self):
        print(f"\n--------------- Analysing record times:")
        self._analyse_records_timing()

        print(f"\n--------------- Analysing detections:")
        self._analyse_detections()


    def _analyse_records_timing(self):
        """
        analyse just message timing with / without detections
        """

        print('kmd7 raw records (with / without detections): ')
        raw_records_dt = None
        if len(self.raw_record_times) > 0:
            raw_records_dt = np.array(self.raw_record_times[1:]) - np.array(self.raw_record_times[:-1])
            step_mean = np.mean(raw_records_dt)
            step_std = np.std(raw_records_dt)
            print('     {} records'.format(len(self.raw_record_times)))
            print('     start: {:.4f}'.format(self.raw_record_times[0]))
            print('     duration: {:.4f}[sec]'.format(self.raw_record_times[-1] - self.raw_record_times[0]))
            print('     time steps: {:.2f}+-{:.3f}[sec]  (mean, std)'.format(step_mean, step_std))

        print('kmd7 track records (with / without detections):')
        track_records_dt = None
        if len(self.track_record_times) > 0:
            track_records_dt = np.array(self.track_record_times[1:]) - np.array(self.track_record_times[:-1])
            step_mean = np.mean(track_records_dt)
            step_std = np.std(track_records_dt)
            print('     {} records'.format(len(self.track_record_times)))
            print('     start: {:.4f}'.format(self.track_record_times[0]))
            print('     duration: {:.4f}[sec]'.format(self.track_record_times[-1] - self.track_record_times[0]))
            print('     time steps: {:.2f}+-{:.3f}[sec]  (mean, std)'.format(step_mean, step_std))

        fig = plt.figure('bag record timing')
        ax1 = plt.Subplot(fig, 211)
        fig.add_subplot(ax1)
        ax2 = plt.Subplot(fig, 212)
        fig.add_subplot(ax2)
        fig.suptitle('time difference between records', fontsize=15)

        if raw_records_dt is not None:
            ax1.scatter(range(0, raw_records_dt.size), raw_records_dt, color=(0, 0, 1), s=2, alpha=1)
            ax1.set_xlabel(r'index', fontsize=12)
            ax1.set_ylabel(r'time difference [sec]', fontsize=12)
            ax1.set_title('raw records times', fontsize=14)
            ax1.grid(True)

        if track_records_dt is not None:
            ax2.scatter(range(0, track_records_dt.size), track_records_dt, color=(0, 0, 1), s=2, alpha=1)
            ax2.set_xlabel(r'index', fontsize=12)
            ax2.set_ylabel(r'time difference [sec]', fontsize=12)
            ax2.set_title('track records times', fontsize=14)
            ax2.grid(True)

        fig.tight_layout()
        plt.pause(0.1)
        plt.show(block=True)

        print('\n')
        return


    def _analyse_detections(self):
        """
        analyse just message timing with / without detections
        """

        print('kmd7 raw records (with / without detections): ')
        raw_records_dt = None

        if len(self.raw_detections) > 0:

            filtered_detections = [d for d in self.raw_detections if d['velocity'] >= -45 and d['velocity'] <= -35 and d['range'] > 5   and d['range'] < 40]

            # gather records by time
            records = {}
            for d in filtered_detections:
                if d['time'] not in records.keys():
                    records[d['time']] = []
                records[d['time']].append(d)


            #----------- histogram of number of detections per time ---------------------
            detection_times = np.array([d['time'] for d in filtered_detections])
            value_counts = Counter(detection_times)

            # Count how many values occur exactly n times
            max_n = 20
            hist_counts = np.zeros(max_n, dtype=int)  # index 0 -> n=1
            for count in value_counts.values():
                if 1 <= count <= max_n:
                    hist_counts[count - 1] += 1  # count=1 goes to index 0

            fig = plt.figure('raw detection records')
            ax1 = fig.add_subplot(1, 1, 1)  # 1 row, 1 col, first subplot
            ax1.bar(range(1, max_n + 1), hist_counts, color='skyblue', edgecolor='black')
            ax1.set_xlabel("Number of detection per time")
            ax1.set_ylabel("Number of times with n detections")
            ax1.set_title("number of detections per time")
            ax1.set_xticks(range(1, max_n + 1))
            plt.show(block=False)

            # print('raw detections:')
            # t0 = min(records.keys())
            # for r in records:
            #     print('  record time {:.3f}'.format(r-t0))
            #     for d in records[r]:
            #         print('  {}'.format(d))


            #----------- range Vs velocity ---------------------
            rv = np.array([[d['range'], d['velocity']] for d in filtered_detections])
            fig = plt.figure('range Vs velocity')
            ax1 = fig.add_subplot(1, 1, 1)  # 1 row, 1 col, first subplot
            ax1.scatter(rv[:, 0], rv[:, 1])
            ax1.set_xlabel("range")
            ax1.set_ylabel("velocity")
            ax1.set_title("range Vs velocity")
            plt.show(block=False)


            #----------- range velocity vs time ---------------------
            t = np.array([d['time'] for d in filtered_detections])
            fig = plt.figure('range and velocity Vs time')
            ax1 = fig.add_subplot(2, 1, 1)  # 1 row, 1 col, first subplot
            ax1.scatter(t, rv[:, 0])
            ax1.set_xlabel("time")
            ax1.set_ylabel("range")
            ax1.set_title("range Vs time")

            ax2 = fig.add_subplot(2, 1, 2)  # 1 row, 1 col, first subplot
            ax2.scatter(t, rv[:, 1])
            ax2.set_xlabel("time")
            ax2.set_ylabel("velocity")
            ax2.set_title("velocity Vs time")

            plt.show(block=False)

        # Example data: columns = [range, velocity]
        fig, ax = plt.subplots(figsize=(7, 5))
        scat = ax.scatter([], [], color='red')
        ax.set_xlim(rv[:, 0].min() - 5, rv[:, 0].max() + 5)
        ax.set_ylim(rv[:, 1].min() - 5, rv[:, 1].max() + 5)
        ax.set_xlabel("Range")
        ax.set_ylabel("Velocity")

        record_times = sorted(records.keys())
        for i, t in enumerate(record_times):
            x = [d['range'] for d in records[t]]
            y = [d['velocity'] for d in records[t]]
            scat.set_offsets(np.c_[x, y])
            # Update title dynamically
            ax.set_title(f"Range vs Velocity - Time step {i}")
            # pause and draw
            plt.pause(0.2)  # pause for animation
            fig.canvas.draw()

        plt.show()

        return

    def _get_kmd7_raw(self):
        """
        get kmd7 raw data
        """

        # get image messages
        self.raw_record_times = []
        self.raw_detections = []
        connections = [x for x in self.bag.connections if x.topic == self.kmd7_raw_topic]
        for connection, timestamp, rawdata in self.bag.messages(connections=connections):
            msg = self.bag.deserialize(rawdata, connection.msgtype)
            if msg.__msgtype__ == self.kmd7_raw_msg_type:
                msg_time = self.__to_time(msg.header.stamp)
                self.raw_record_times.append(msg_time)
                if len(msg.detections) > 0:
                    for det in msg.detections:
                        self.raw_detections.append({'time': msg_time,
                                           'range': det.range,          # [m]
                                           'velocity': det.velocity,    # [m/s]
                                           'amplitude': det.amplitude,  # [DB]
                                           'azimuth': det.azimuth       # [deg] + is right
                                                    })

        # check if sorted
        if len(self.raw_record_times) > 0:
            is_sorted = np.all(np.sort(self.raw_record_times) == self.raw_record_times)
            if not is_sorted:
                raise Exception('kmd7_raw times record not sorted!')
        if len(self.raw_detections) > 0:
            detection_times = [d['time'] for d in self.raw_detections]
            is_sorted = np.all(np.sort(detection_times) == detection_times)
            if not is_sorted:
                raise Exception('kmd7_raw detection times not sorted!')
        return

    def _get_kmd7_track(self):
        """
        get kmd7 raw data
        """

        # get image messages
        self.track_record_times = []
        self.track_detections = []
        connections = [x for x in self.bag.connections if x.topic == self.kmd7_track_topic]
        for connection, timestamp, rawdata in self.bag.messages(connections=connections):
            msg = self.bag.deserialize(rawdata, connection.msgtype)
            if msg.__msgtype__ == self.kmd7_track_msg_type:
                msg_time = self.__to_time(msg.header.stamp)
                self.track_record_times.append(msg_time)
                if len(msg.detections) > 0:
                    for det in msg.detections:
                        pass
                        # self.track_detections.append({'time': msg_time,
                        #                    'range': det.range,          # [m]
                        #                    'velocity': det.velocity,    # [m/s]
                        #                    'amplitude': det.amplitude,  # [DB]
                        #                    'azimuth': det.azimuth       # [deg] + is right
                        #                             })

        # check if sorted
        if len(self.track_record_times) > 0:
            is_sorted = np.all(np.sort(self.track_record_times) == self.track_record_times)
            if not is_sorted:
                raise Exception('kmd7_raw times record not sorted!')
        if len(self.track_detections) > 0:
            detection_times = [d['time'] for d in self.track_detections]
            is_sorted = np.all(np.sort(detection_times) == detection_times)
            if not is_sorted:
                raise Exception('kmd7_raw detection times not sorted!')
        return

    @staticmethod
    def __to_time(ros_time_stamp):
        return ros_time_stamp.sec + ros_time_stamp.nanosec * 1e-9

    def save(self, output_folder, start_time=None, end_time=None, detection_polygon_output_file=None):
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

            # if detection_polygon_output_file is not None:
            #     self._save_detection_input_polygon(detection_polygon_output_file)


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


                        # get detection input polygon
                        if draw_detection_polygon:
                            if np.abs(frame_time - 1749395144.490971) < 0.05:
                                aa = 5
                            polygon_points = self._get_closest_detection_polygon(frame_time, frames_time_step)
                            polygon_points = polygon_points[0]
                            if polygon_points is not None:
                                pp = np.round(polygon_points).astype(np.int32)
                                cv_img = cv2.polylines(cv_img, np.array([pp]), isClosed=True,
                                                       color=(255, 0, 0), thickness=1)

                        # get detection results
                        if draw_detection_results:
                            bbox_points = self._get_closest_detection_results(frame_time, frames_time_step)
                            bbox_points = bbox_points[0]
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
                        detection_polygon = self._get_closest_detection_polygon(frame_time, frames_time_step)
                        if detection_polygon[0] is not None:
                            detection_polygon[0] = detection_polygon[0].tolist()
                        frame_polygons.set(frame_id_left, detection_polygon[0])
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
            polygon_data.set(i, polygon_points)
        polygon_data.save(output_file)
        return

    def __del__(self):
        if type(self.bag) == AnyReader and self.bag.isopen:
            self.bag.close()

if __name__ == '__main__':


    # ------------------ kfar_masarik 08.06.2025 ------------------------------

    base_folder = '/home/roee/Projects/datasets/interceptor_drone/20260107_reshafim'
    # bag_folder = os.path.join(base_folder, '20260107_1311_46/20260107_1312_48/drone_20260107_1413')  # short scenario
    bag_folder = os.path.join(base_folder, '20260107_1311_46/20260107_1313_58/drone_20260107_1414')  # good scenario
    # bag_folder = os.path.join(base_folder, '20260107_1311_46/20260107_1314_52/drone_20260107_1415')  # crash scenario (ends early)
    kmd7_raw_topic = '/kmd7_node/raw_detections'
    kmd7_track_topic = '/kmd7_node/tracked_detections'
    valid_record_times = {'start': -np.inf, 'end': np.inf}
    output_file = os.path.join(bag_folder, 'kmd7_data.json')

    # analyse and sync record
    ros_record = RosBagRecord(bag_folder, kmd7_raw_topic=kmd7_raw_topic, kmd7_track_topic=kmd7_track_topic)
    ros_record.analyse_bag()
    t1 = time.monotonic()
    ros_record.save(output_file, start_time=valid_record_times['start'], end_time=valid_record_times['end'])
    print('save_to_folder time - {}sec'.format(time.monotonic()-t1))
    print('Done!')
