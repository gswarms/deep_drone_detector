"""Extract information from a video. """

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')


class VideoRecord:
    """
    This object converts video record to folder standard record
    """
    def __init__(self, video_file):

        self.frames = None
        self.start_time = None
        self.end_time = None

        self.video_file = video_file
        if (not os.path.isfile(self.video_file)):
            raise Exception('video file {} not found!'.format(video_file))
        self.cap = None

        self.frame_times = None
        self.ref_bbox_times = None
        return

    def save_to_folder(self, output_folder, start_time=None, end_time=None):
        """
        save camera frames and to standard record folder format
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
        timestamps_file = os.path.join(output_folder, 'timestamps.txt')

        image_subfolder_name = os.path.join(output_folder, 'images')
        if not os.path.isdir(image_subfolder_name):
            os.makedirs(image_subfolder_name)

        self.cap = cv2.VideoCapture(self.video_file)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        video_time_step = 1/fps

        # Check if camera opened successfully
        if (self.cap.isOpened() == False):
            print("Error opening video file")


        with open(timestamps_file, 'w') as f:
            f.write('#timestamp [ns]    filename\n')

            # Read until video is completed
            frame_id = 0
            frame_time = 0
            timestamps_data = []
            while self.cap.isOpened():
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                if ret == True:
                    # # Display the resulting frame
                    # cv2.imshow('Frame', frame)
                    #
                    # # Press Q on keyboard to exit
                    # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #     break

                    if self.start_time <= frame_time <= self.end_time:
                        image_file_name = '{:06d}.png'.format(frame_id)
                        image_file_path = os.path.join(image_subfolder_name, image_file_name)
                        cv2.imwrite(image_file_path, frame)
                        f.write('{:.6f} {}\n'.format(float(frame_time), image_file_name))

                    frame_time = frame_time + video_time_step
                    frame_id = frame_id + 1

                # Break the loop
                else:
                    break

            # When everything done, release
            # the video capture object
            self.cap.release()

        return

    def __del__(self):
        if type(self.cap) == cv2.VideoCapture and self.cap.isOpened():
            self.cap.release()


if __name__ == '__main__':

    video_file = '/home/roee/Projects/datasets/interceptor_drone/20250223_kfar_galim/20250225_exp8_video13.mp4'
    valid_record_times = {'start': 0, 'end': 5}
    output_folder = '/home/roee/Projects/datasets/interceptor_drone/20250223_kfar_galim/20250225_exp8_video13_extracted'

    rec = VideoRecord(video_file)
    rec.save_to_folder(output_folder, start_time=valid_record_times['start'], end_time=valid_record_times['end'])
    print('Done!')