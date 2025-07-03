""" 3D geometry tools
"""
import cv_core.pinhole_camera
import numpy as np
import cv2
from cv_core import pinhole_camera as phc
from cv_core import geometry_2D as g2d
from cv_core import geometry_3D as g3d


class LosPixelConverter:
    def __init__(self):
        """
        This object is used for converting pixels to line of sight and vice versa
        Functionality:
        1) body frame los + los_uncertainty to image polygon (can be used to set_roi_polygon for blob yolo_detector)
        2) pixel to los - convert image pixel to body frame los (can be used to convert tracking results to 3D los)
        3) world to pixel - convert a 3D doby frame world point to an image pixel

        * note: this object mainly wraps cv_core.pinhole camera functionality in a more friendly way to be used by
                 an external program e.g. much easier for python binding

        inputs:
        param: camera_intrinsic_matrix - pinhole camera intrinsic matrix (opencv format)
                                        fx, 0, cx
                                        0, fy, cy
                                        0,  0,  1
        param: distortion_coefficients - pinhole camera distortion coefficients (opencv format)
        param: image_size - (width, height) in pixels
        param: camera_extrinsic_matrix - camera extrinsic matrix:
                                          R3x3  t3x1
                                         0,0,0, 1
                                         This matrix transforms a point from camera to world!
                                         so:
                                          t = camera position in world frame (x, y, z)
                                          R = camera axes in world frame
                                             (| | |
                                              x y z
                                              | | |)
        """
        self.camera = None
        return

    def set_camera(self, camera_intrinsic_matrix, distortion_coefficients, image_size, camera_extrinsic_matrix):
        """
        set pinhole camera parameters

        inputs:
        param: camera_intrinsic_matrix - pinhole camera intrinsic matrix (opencv format)
                                        fx, 0, cx
                                        0, fy, cy
                                        0,  0,  1
        param: distortion_coefficients - pinhole camera distortion coefficients (opencv format)
        param: image_size - (width, height) in pixels
        param: camera_extrinsic_matrix - camera extrinsic matrix:
                                          R3x3  t3x1
                                         0,0,0, 1
                                         This matrix transforms a point from camera to world!
                                         so:
                                          t = camera position in world frame (x, y, z)
                                          R = camera axes in world frame
                                             (| | |
                                              x y z
                                              | | |)
        """
        self.camera = phc.PinholeCamera()
        self.camera.set(id='cam0', model=phc.CameraModel.PINHOLE,
                        intrinsic_matrix=camera_intrinsic_matrix, dist_coeffs=distortion_coefficients,
                        image_size=image_size, T_cam_to_body=camera_extrinsic_matrix,
                        skew=0)
        return


    def image_polygon_from_los(self, los, los_angular_uncertainty, num_points=12, int_polygon_coordinates=False, keep_num_points=False, verbose=False):
        """
        convert los and angular uncertainty in body frame to ROI in camera image

        param: los - 3D line of sight in body frame (x, y, z)
        param: los_angular_uncertainty - los scalar angle uncertainty scalar [radians]
        param: num_points - number of polygon points
        param: int_polygon_coordinates - convert polygon point coordinates to int (and round them)
        param: keep_num_points - keep the set number of points in case of intersecting polygon with image borders.
                                 point will be inserted between polygon points to achieve the required number of points.
        """

        if verbose:
            print('check inputs')

        if self.camera is None:
            raise Exception('camera not set!')
        if num_points <= 6:
            raise Exception('number of points must e >= 6!')

        los = np.array(los, dtype=np.float32)
        if los.size != 3:
            raise Exception('invalid los input!')
        los = los.reshape(3,1)
        los_norm = np.linalg.norm(los)
        if los_norm <= 0:
            raise Exception('invalid los input!')

        # rotate line of sight to camera coordinates
        R = self.camera.T_cam_to_body[:3,:3]
        los_cam = np.matmul(R.T, los).flatten()
        los_cam = los_cam / np.linalg.norm(los_cam)

        if verbose:
            print('find los vectors')

        # Choose a random vector w that is not parallel to los
        w = np.array([1, 0, 0]) if np.dot(los_cam, [1, 0, 0]) <= 0.707 else np.array([0, 1, 0])
        # Project w onto the plane perpendicular to v
        w_proj = w - np.dot(w, los_cam) * los_cam
        # Normalize the projection of w
        w_proj = w_proj / np.linalg.norm(w_proj)
        # Now we have a vector perpendicular to v (w_proj)
        # The desired vector u is a combination of v and w_proj such that the angle between them is theta
        u = np.cos(los_angular_uncertainty) * los_cam + np.sin(los_angular_uncertainty) * w_proj

        # find multiple vectors with the same angle theta from los
        # rotate u around the los
        n = num_points
        a = np.linspace(0, (2 * np.pi) * ((n - 1) / n), n, dtype=np.float32)
        ui = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            rot_vec = los_cam * a[i]
            R, _ = cv2.Rodrigues(rot_vec)
            ui[i, :] = np.matmul(R, u)
            # print('ui angle to los: {}'.format(np.arccos(np.dot(los, ui[i, :]))*180/np.pi))

        if verbose:
            print('los vectors found')
            print('project to image')

        # project to camera image
        body_pose = g3d.rigid3dtform.Rigid3dTform(np.eye(3), (0, 0, 0))
        image_points, is_in_image = self.camera.project_points(ui, body_pose)
        is_in_front = ui[:,2] > 0
        image_points = image_points[is_in_front, :]
        is_in_image = is_in_image[is_in_front]

        if not np.any(is_in_image):
            valid_polygon_points = np.zeros((0, 2))
            return valid_polygon_points

        if verbose:
            print('projected to image')
            print('intersect with image borders')
        image_borders = np.array(((0, 0),
                                  (self.camera.image_size[0], 0),
                                  (self.camera.image_size[0], self.camera.image_size[1]),
                                  (0, self.camera.image_size[1])), dtype=np.float32)

        # img = np.zeros((self.camera.image_size[1], self.camera.image_size[0], 3), dtype=np.uint8) + 255
        # image_points_int = image_points.astype(int)
        # img = cv2.polylines(img, [image_points_int], isClosed=True, color=(100, 255, 100), thickness=1)

        valid_polygon_points = g2d.polygon_2D.polygon_intersect(image_points, image_borders, use_shapely=True)
        if valid_polygon_points.shape[0] == image_points.shape[0] + 1:
            valid_polygon_points = valid_polygon_points[:-1,:2]  # remove last point. It is a duplicate of the first because shapely returns close polygons.

        if verbose:
            print('intersected with image borders')
            print('check validity')

        if valid_polygon_points.shape[0] < 3:
            if verbose:
                print('outside of image borders or <3 points - return empty')
            valid_polygon_points = np.zeros((0, 2))

        else:
            if keep_num_points:
                if verbose:
                    print('adjust number of polygon points {}-{}'.format(valid_polygon_points.shape[0], num_points))
                valid_polygon_points = polygon_adjust_number_of_points(valid_polygon_points, num_points)

        if int_polygon_coordinates:
            if verbose:
                print('convert to int')
            valid_polygon_points = np.round(valid_polygon_points).astype(np.int32)

        return valid_polygon_points

    def pixel_to_los(self, image_points):
        """
        convert image pixel to los in body frame

        param: image_points - (nx2)
        """
        if self.camera is None:
            raise Exception('camera not set!')

        R = self.camera.T_cam_to_body[:3,:3]
        image_points = np.array(image_points).reshape(-1,2)
        los, is_in_image = self.camera.pixel_to_los(image_points, R)

        return los, is_in_image

    def pixel_to_los_single(self, image_point_x, image_point_y):
        """
        convert a single image pixel to los in body frame
        This flavor is specifically made to support a specific implementation

        param: image_point_x - image pixel x
        param: image_point_x - image pixel y
        """
        if self.camera is None:
            raise Exception('camera not set!')

        R = self.camera.T_cam_to_body[:3,:3]
        image_points = np.array((image_point_x, image_point_y)).reshape(-1,2)
        los, is_in_image = self.camera.pixel_to_los(image_points, R)

        return list(los.flatten())

    def project_points(self, world_points):
        """
        convert world point to image pixel

        param: los - (nx3)
        """
        if self.camera is None:
            raise Exception('camera not set!')

        R = self.camera.T_cam_to_body[:3, :3]
        t = self.camera.T_cam_to_body[:3, 3]
        T = phc.Rigid3dTform(R, t)
        image_points = self.camera.project_points(world_points, T)
        return image_points

    @staticmethod
    def calc_target_los_angular_uncertainty(prior_angular_uncertainty=0, relative_position_uncertainty=None,
                                            range_to_target=None, angular_uncertainty_limit=None):
        """
        see calc_target_los_angular_uncertainty
        This doesn't really belong to this class. This method is here to make implementation easier.
        """
        return calc_target_los_angular_uncertainty(prior_angular_uncertainty, relative_position_uncertainty,
                                                   range_to_target, angular_uncertainty_limit)


def polygon_adjust_number_of_points(polygon_points, required_num_points):
    """
    add / remove points from a polygon to get a required number of points.

    algorithms:
    adding points:
       add point in the middle of the longest edge iteratively.
       This keeps the same polygon shape (with more points)

    removing points:
        common approach is Ramer–Douglas–Peucker (RDP) Algorithm
        We don't use this because:
        1. This requires uses shapely
        2. It's hard to control the exact number of output vertices
        3. no assurance the result polygon is completely inclosed in the input polygon, so this might be a problem if some of the polygon is image borders.

        We just iteratively remove the smallest 3-vertex triangle.

    param: polygon_points - polygon points (nx2). must be ordered clockwise or anti-clockwise
    param: required_num_points - number of required points
    """

    polygon_points = np.array(polygon_points)
    if polygon_points.shape[1] != 2:
        raise Exception('invalid polygon shape. must be (nx2)!')
    if required_num_points < 3:
        raise Exception('invalid number of required points. must be >= 3!')

    n = polygon_points.shape[0]
    if required_num_points == n:
        res_polygon = polygon_points  # don't need to change anything

    elif required_num_points > n:
        res_polygon = polygon_points
        n_missing = required_num_points - n
        for i in range(n_missing):
            # find the longest edge
            shifted = np.roll(res_polygon, -1, axis=0)
            d = np.linalg.norm(res_polygon - shifted, axis=1)
            idx = np.argmax(d)
            # add a point in the middle of the longest edge
            m = res_polygon.shape[0]
            p = (res_polygon[idx, :] + res_polygon[(idx + 1) % m, :]) / 2
            # res_polygon = np.vstack((res_polygon[:idx + 1, :], p, res_polygon[idx + 1:]))
            res_polygon = np.insert(res_polygon, (idx+1)%m, p, axis=0)

    else:  # required_num_points < n
        res_polygon = polygon_points
        n_remove = n - required_num_points

        tri_area = np.zeros(n, dtype=np.float32)
        for i in range(n):
            a = np.array(res_polygon[(i - 1) % n])
            b = np.array(res_polygon[i % n])
            c = np.array(res_polygon[(i + 1) % n])
            tri_area[i] = triangle_area(a, b, c)

        for i in range(n_remove):
            # find the smallest triangle, and remove central vertex
            idx = np.argmin(tri_area)

            # fix neighboring triangle area calculation
            m = res_polygon.shape[0]
            a1 = np.array(res_polygon[(idx - 1) % m])
            b1 = np.array(res_polygon[(idx + 1) % m])
            c1 = np.array(res_polygon[(idx + 2) % m])
            tri_area[i+1 % m] = triangle_area(a1, b1, c1)

            a2 = np.array(res_polygon[(idx - 2) % m])
            b2 = np.array(res_polygon[(idx - 1) % m])
            c2 = np.array(res_polygon[(idx + 1) % m])
            tri_area[i-1 % m] = triangle_area(a2, b2, c2)

            # pop
            res_polygon = np.delete(res_polygon, idx, axis=0)
            tri_area = np.delete(tri_area, idx, axis=0)

    return res_polygon

def triangle_area(a, b, c):
    """
    calc a triangle area
    a, b, c - triangle corners (2,1) each
    """
    return 0.5 * abs((a[0] * (b[1] - c[1]) +
                      b[0] * (c[1] - a[1]) +
                      c[0] * (a[1] - b[1])))


def calc_target_los_angular_uncertainty(prior_angular_uncertainty=0, relative_position_uncertainty=None,
                                        range_to_target=None, angular_uncertainty_limit=None):
    """
    calc angular uncertainty for the line of sight to some target

    :param prior_angular_uncertainty - prior angular uncertainty in [rad]
    :param relative_position_uncertainty - relative position uncertainty in [m]
    :param range_to_target - range between us and the target in [m]
    :param angular_uncertainty_limit - top limit to total error [rad]. Usually it makes sense to limit to 90 deg.
    """

    if relative_position_uncertainty is None and range_to_target is None:
        position_induced_uncertainty = 0
    elif relative_position_uncertainty is not None and range_to_target is not None:
        position_induced_uncertainty = np.arctan2(np.abs(relative_position_uncertainty), range_to_target)
    else:
        raise Exception('invalid input! relative_position_uncertainty and range_to_target must be set together!')

    if angular_uncertainty_limit is not None:
        res = min(float(prior_angular_uncertainty) + float(position_induced_uncertainty), angular_uncertainty_limit)  # can't be more than 90 degrees
    else:
        res = float(prior_angular_uncertainty) + float(position_induced_uncertainty)
    return res


if __name__ == '__main__':

    image_file = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-35-10/camera_2025_7_1-6_35_13_extracted/images/000.png'
    camera_calibration_file = '/home/roee/Projects/datasets/interceptor_drone/20250612_calibration/20250612_pz001_calibration/camera_intrinsics_IDC1.yaml'

    los = np.array([[0,0,1]])
    los_angular_uncertainty = 10*np.pi/180

    # load image
    img = cv2.imread(image_file)

    # load camera
    cam = cv_core.pinhole_camera.PinholeCamera()
    cam.load(camera_calibration_file)
    camera_intrinsic_matrix = cam.K
    distortion_coefficients = cam.distortion_coefficients
    image_size = cam.image_size
    camera_extrinsic_matrix = np.eye(4)  # cam.T_cam_to_body

    lpc =  LosPixelConverter()
    lpc.set_camera(camera_intrinsic_matrix, distortion_coefficients, image_size, camera_extrinsic_matrix)

    roi_polygon =  lpc.image_polygon_from_los(los, los_angular_uncertainty, num_points=12, int_polygon_coordinates=False, keep_num_points=False, verbose=False)

    roi_polygon = np.round(roi_polygon).astype(np.int32)
    points = roi_polygon.reshape((-1, 1, 2))


    cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.imshow('image with roi polygon', img)
    cv2.waitKey(50)

    print('done!')
    cv2.destroyAllWindows()