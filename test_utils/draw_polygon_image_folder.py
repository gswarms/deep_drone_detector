"""
draw polygon on images
"""
import os
import cv2
import numpy as np
import roi_utils
import glob

# ***** global variable declaration *****
done = False
points = []
current = (0, 0)
prev_current = (0, 0)
img_tmp = None


def on_mouse(event, x, y, buttons, user_param):
    global done, points, current, img_tmp
    # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
    if done:  # Nothing more to do
        return
    if event == cv2.EVENT_MOUSEMOVE:
        # We want to be able to draw the line-in-progress, so update current mouse position
        current = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Left click means adding a point at current position to the list of points
        print("Adding point #%d with position(%d,%d)" % (len(points), x, y))
        cv2.circle(img_tmp, (x, y), 5, (0, 200, 0), -1)
        points.append([x, y])
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # Right click means we're done
        print("Completing polygon with %d points." % len(points))
        done = True


if __name__ == '__main__':

    # ***** replace with required image path *****
    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250519_083827_extracted/images'
    # polygon_file = '/home/roee/Projects/datasets/interceptor_drone/20250511_kfar_galim/camera_20250519_083827_extracted/kfar_galim_20250511_133041_polygons.yaml'

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250519_kfar_galim/camera_20250519_083827_extracted/images'
    # polygon_file = '/home/roee/Projects/datasets/interceptor_drone/20250519_kfar_galim/camera_20250519_083827_extracted/kfar_galim_20250519_083827_polygons.yaml'

    # images_folder = '/home/roee/Downloads/camera_2025_6_5-12_56_26_extracted/images'
    # polygon_file = '/home/roee/Downloads/camera_2025_6_5-12_56_26_extracted/kfar_galim_20250605_125626_polygons.yaml'

    # images_folder = '/home/roee/Downloads/camera_2025_6_6-3_0_31_extracted/images'
    # polygon_file = '/home/roee/Downloads/camera_2025_6_6-3_0_31_extracted/kfar_galim_20250606_030031_polygons.yaml'

    # images_folder = '/home/roee/Downloads/camera_2025_6_5-11_47_39_extracted/images'
    # polygon_file = '/home/roee/Downloads/camera_2025_6_5-11_47_39_extracted/kfar_galim_20250605_114739_polygons.yaml'

    # ------------------ kfar massarik 08.06.2025 ------------------------------
    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-04-53/camera_2025_6_8-15_4_56_extracted/images'
    # polygon_file = os.path.join(images_folder,'..','20250608_180453_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-29-51/camera_2025_6_8-15_29_54_extracted/images'
    # polygon_file = os.path.join(images_folder,'..','20250608_182951_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-51-15/camera_2025_6_8-15_51_18_extracted/images'
    # polygon_file = os.path.join(images_folder, '..', '20250608_185115_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-52-56/camera_2025_6_8-15_52_58_extracted/images'
    # polygon_file = os.path.join(images_folder, '..', '20250608_185256_manual_detection_roi_polygons.yaml')
    #
    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-53-46/camera_2025_6_8-15_53_49_extracted/images'
    # polygon_file = os.path.join(images_folder, '..', '20250608_185346_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_18-59-46/camera_2025_6_8-15_59_49_extracted/images'
    # polygon_file = os.path.join(images_folder, '..', '20250608_185946_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-00-25/camera_2025_6_8-16_0_28_extracted/images'
    # polygon_file = os.path.join(images_folder, '..', '20250608_190025_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-08-34/camera_2025_6_8-16_8_48_extracted/images'
    # polygon_file = os.path.join(images_folder, '..', '20250608_190834_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250608_kfar_masarik/2025-06-08_19-25-35/camera_2025_6_8-16_25_38_extracted/images'
    # polygon_file = os.path.join(images_folder, '..', '20250608_192535_manual_detection_roi_polygons.yaml')


    # ------------------ kfar galim 01.07.2025 ------------------------------
    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_08-26-13/camera_2025_7_1-5_26_17_extracted/images'
    # polygon_file = os.path.join(images_folder,'..','20250701_082613_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-01-54/camera_2025_7_1-6_2_3_extracted/images'
    # polygon_file = os.path.join(images_folder,'..','20250701_090154_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-27-36/camera_2025_7_1-6_27_39_extracted/images'
    # polygon_file = os.path.join(images_folder,'..','20250701_092736_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-34-06/camera_2025_7_1-6_34_9_extracted/images'
    # polygon_file = os.path.join(images_folder,'..','20250701_093406_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-35-10/camera_2025_7_1-6_35_13_extracted/images'
    # polygon_file = os.path.join(images_folder,'..','20250701_093510_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-46-51/camera_2025_7_1-6_46_54_extracted/images'
    # polygon_file = os.path.join(images_folder,'..','20250701_094651_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-48-20/camera_2025_7_1-6_48_22_extracted'  # bad
    # polygon_file = os.path.join(images_folder, '..', '20250701_???_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-03-29/camera_2025_7_1-7_3_32_extracted/images'
    # polygon_file = os.path.join(images_folder,'..','20250701_100329_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-04-48/camera_2025_7_1-7_4_52_extracted/images'  # bad
    # polygon_file = os.path.join(images_folder, '..', '20250701_???_manual_detection_roi_polygons.yaml')

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-47-42/camera_2025_7_1-7_47_54_extracted/images'  # bad
    # polygon_file = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-49-24/camera_2025_7_1-7_49_27'

    # images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-47-42/camera_2025_7_1-7_47_54_extracted/images'  # bad
    # polygon_file = os.path.join(images_folder, '..', '20250701_???_manual_detection_roi_polygons.yaml')

    images_folder = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_10-49-24/camera_2025_7_1-7_49_27_extracted/images'
    polygon_file = os.path.join(images_folder, '..', '20250701_104924_manual_detection_roi_polygons.yaml')


    image_resize = None  # (640, 480)
    image_file_suffix = 'png'

    polygon_num_points = 4
    annotate_polygons = True
    show_polygons = True

    if annotate_polygons:
        video_time_step = 1

        # Check images folder
        if os.path.isdir(images_folder):
            print('annotating images dir {}...'.format(images_folder))
        else:
            raise Exception('images dir {} not found!'.format(images_folder))

        # setup window
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_mouse)

        frame_polygons = []
        done = False
        frame_id = 0

        image_files = glob.glob(os.path.join(images_folder, '*.'+image_file_suffix))
        image_files = sorted(image_files)

        if image_resize is None:
            frame = cv2.imread(image_files[0])
            frame_size = (frame.shape[1], frame.shape[0])
        else:
            frame_size = image_resize
        frame_polygons = roi_utils.PolygonPerFrame(frame_size=frame_size)

        for imfile in image_files:

            # Capture frame-by-frame
            frame = cv2.imread(imfile)
            ret = frame is not None

            if ret == True:
                if image_resize is not None:
                    frame = cv2.resize(frame, image_resize)

                cv2.imshow("image", frame)
                cv2.putText(frame, 'frame {}'.format(frame_id), (20,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,   color=(100, 255, 255))
                cv2.waitKey(100)
                key = cv2.waitKey(0)
                # wait for user key:
                # - space means proceed
                # - esc means stop marking
                # - anything else mean mark this frame

                if key == ord('\x1b'):  # press ESC to quit
                    break

                elif key == ord('p'):  # press 'p' to mark a polygon
                    print('set polygon frame')
                    done = False
                    points = []
                    current = (0, 0)
                    prev_current = None
                    img_tmp = frame.copy()

                    while not done:
                        # This is our drawing loop, we just continuously draw new images
                        # and show them in the named window
                        if len(points) > 0:
                            if current != prev_current:
                                frame = img_tmp.copy()
                                prev_current = current
                            # Draw all the current polygon segments
                            cv2.polylines(frame, [np.array(points)], False, (255, 0, 0), 1)
                            # And  also show what the current segment would look like
                            cv2.line(frame, (points[-1][0], points[-1][1]), current, (0, 0, 255))

                        # Update the window
                        cv2.imshow("image", frame)
                        cv2.waitKey(50)

                    # User finished entering the polygon points, so let's make the final drawing
                    # of a filled polygon
                    if len(points) > 0:
                        image = cv2.polylines(frame, np.array([points]),
                                              isClosed=True, color=(255, 0, 0), thickness=3)
                    # And show it
                    cv2.imshow("image", frame)
                    # Waiting for the user to press any key
                    key = cv2.waitKey(50)

                    if len(points) == polygon_num_points:
                        frame_polygons.set(frame_id, points)
                    else:
                        print('Warning! invalid number of polygon points! got {} expecting {}! skipping this frame.'.format(len(points), polygon_num_points))

                else:
                    print('skip frame')

                frame_id = frame_id + 1
            else:
                break

        # add interpolated polygons to all frames
        num_frames = len(image_files)
        frame_ids = range(num_frames)
        frame_polygons.interpolate_polygons(frame_ids)

        frame_polygons.save(polygon_file)
        cv2.destroyWindow("image")


    if show_polygons:
        frame_polygons2 = roi_utils.PolygonPerFrame(frame_size=None)
        frame_polygons2.load(polygon_file)

        cv2.namedWindow("image")

        # draw polygons on images
        frame_id = 0
        for imfile in image_files:
            # Capture frame-by-frame
            frame = cv2.imread(imfile)
            ret = frame is not None
            if ret == True:
                if image_resize is not None:
                    frame = cv2.resize(frame, image_resize)
                pts = frame_polygons2.get(frame_id)
                if pts is not None:
                    pts = np.array(pts, dtype = np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 255))
                cv2.imshow("image", frame)
                cv2.waitKey(100)
                frame_id = frame_id + 1

            else:
                break

        cv2.destroyWindow("image")

    print('Done!')
