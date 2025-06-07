"""
draw polygon on video frames
"""
import cv2
import numpy as np
import roi_utils


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
    video_file = '../examples/drone_example1.mp4'
    polygon_file = '../examples/drone_example1_polygons.yaml'

    image_resize = (960, 540)

    polygon_num_points = 4
    annotate_polygons = True
    show_polygons = True

    if annotate_polygons:
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps !=0 :
            video_time_step = 1 / fps
        else:
            video_time_step = 1

        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video file")

        # setup window
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_mouse)

        if not cap.isOpened():
            print("Error opening video file")

        frame_polygons = []
        done = False
        frame_id = 0
        frame_polygons = roi_utils.PolygonPerFrame(frame_size=image_resize)
        while cap.isOpened():

            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:
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
        cap_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = range(cap_num_frames)
        frame_polygons.interpolate_polygons(frame_ids)
        cap.release()

        frame_polygons.save(polygon_file)
        cv2.destroyWindow("image")


    if show_polygons:
        frame_polygons2 = roi_utils.PolygonPerFrame(frame_size=None)
        frame_polygons2.load(polygon_file)

        cv2.namedWindow("image")

        # draw polygons on images
        cap = cv2.VideoCapture(video_file)
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
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
