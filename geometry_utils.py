""" 3D geometry tools
"""
import copy

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

    def load_camera(self, camera_params_file):
        """
        set pinhole camera parameters

        inputs:
        param: camera_params_file - path to camera parameters yaml file
                                    the YAML file follows the cv_core camera params file format:

                                     id: IDC06
                                     model: PINHOLE
                                     focal_length: [571.15, 570.79]
                                     principal_point: [308.76, 253.59]
                                     image_size: [640, 480]
                                     distortion_coefficients: [-0.2216407, 0.0176510, -0.0009292, -0.0011879, 0.0]
                                     skew: 0
        """
        self.camera = phc.PinholeCamera(camera_params_file)

        # self.camera.set(id='cam0', model=phc.CameraModel.PINHOLE,
        #                 intrinsic_matrix=camera_intrinsic_matrix, dist_coeffs=distortion_coefficients,
        #                 image_size=image_size, T_cam_to_body=camera_extrinsic_matrix,
        #                 skew=0)

        return

    def image_polygon_from_los(self, los, los_angular_uncertainty, num_points=12, int_polygon_coordinates=False,
                               keep_num_points=False, verbose=False):
        """
        project target position uncertainty to camera image polygon

        Important assumptions:
        1. Ignore camera-body position offset (approximate it to be much smaller than range to target).
        2. Assume camera fov is smaller than 180 deg

        Algorithm:
        1. Uniformly sample pixels from the image
        2. project each pixel tp los
        3. test which los are in the cone, and keep the set of valid pixels
        4. Convex hull
        5. Intersect projected polygon with image borders
        6. Simplify polygon to get required number of points

        param: target_position - 3D target mean position in body frame (x, y, z)
        param: target_position_covariance - 3D target position covariance [3x3] in body frame
        param: num_polygon_points - number of polygon_points
        param: num_sample_points - number of sample points
        """

        # TODO: we still have a problem when los_angular_uncertainty > pi/2 and los is to the back.
        #       valid / invalid image points are correct
        #       but using convex hull might be wrong since there is an invalid polygon in the middle of the frame, and valid perimiter!

        if self.camera is None:
            raise Exception('camera not set!')
        if num_points <= 6:
            raise Exception('number of points must e >= 6!')

        los = np.array(los, dtype=np.float32)
        if los.size != 3:
            raise Exception('invalid los input!')
        los = los.reshape(3, 1)
        los_norm = np.linalg.norm(los)
        if los_norm <= 0:
            raise Exception('invalid los input!')

        # ------------- rotate line of sight to camera coordinates -------------
        R = self.camera.T_cam_to_body[:3, :3]
        los_cam = np.matmul(R.T, los).flatten()
        los_cam = los_cam / np.linalg.norm(los_cam)

        # ------------- sample image pixels -------------
        n_rows = 50
        n_cols = 50
        row_indices = np.linspace(0, self.camera.image_size[0] - 1, n_rows, dtype=int)
        col_indices = np.linspace(0, self.camera.image_size[1] - 1, n_cols, dtype=int)
        rows, cols = np.meshgrid(row_indices, col_indices, indexing='ij')
        image_points = np.hstack((rows.reshape(-1,1), cols.reshape(-1,1)), dtype=np.float32)

        # ------------- pixel to los -------------
        Rc = np.eye(3)
        los, is_in_image = self.camera.pixel_to_los(image_points, Rc)
        los_norms = np.linalg.norm(los, axis=1, keepdims=True)
        los_normalized = los / los_norms

        # ------------- test if each los is in the polygon -------------
        cos_angle_to_target = np.dot(los_normalized, los_cam)
        cos_th = np.cos(los_angular_uncertainty)
        is_in_cone = cos_angle_to_target > cos_th

        # ------------- keep only image points that are in the cone -------------
        image_points_valid = image_points[is_in_cone, :]

        # ------------- convex hull -------------
        if image_points_valid.shape[0] < 3:
            if verbose:
                print('outside of image borders or <3 points - return empty')
            valid_polygon_points = np.zeros((0, 2))
        else:
            valid_polygon_points = cv2.convexHull(image_points_valid)
            valid_polygon_points = valid_polygon_points.squeeze()
            valid_polygon_points = polygon_adjust_number_of_points(valid_polygon_points, num_points)

        if int_polygon_coordinates:
            if verbose:
                print('convert to int')
            valid_polygon_points = np.round(valid_polygon_points).astype(np.int32)

        return valid_polygon_points

    # def image_polygon_from_los_orig(self, los, los_angular_uncertainty, num_points=12, int_polygon_coordinates=False, keep_num_points=False, verbose=False):
    #     """
    #     project target position uncertainty to camera image polygon
    #
    #     Note: we ignore camera-body position offset (approximate it to be much smaller than range to target).
    #
    #     Algorithm:
    #     1. Uniformly sample N 3D points on los uncertainty cone
    #     2. Project 3D cone points to camera image pixels
    #        - ignore points behind the camera
    #     3. Convex hull
    #     3. Intersect projected polygon with image borders
    #
    #     param: target_position - 3D target mean position in body frame (x, y, z)
    #     param: target_position_covariance - 3D target position covariance [3x3] in body frame
    #     param: num_polygon_points - number of polygon_points
    #     param: num_sample_points - number of sample points
    #     """
    #
    #     # TODO: there is a problem with this function! rmoving back points is wrong! we need to find the proper solution!
    #
    #     # check inputs
    #     if verbose:
    #         print('check inputs')
    #
    #     if self.camera is None:
    #         raise Exception('camera not set!')
    #     if num_points <= 6:
    #         raise Exception('number of points must e >= 6!')
    #
    #     los = np.array(los, dtype=np.float32)
    #     if los.size != 3:
    #         raise Exception('invalid los input!')
    #     los = los.reshape(3,1)
    #     los_norm = np.linalg.norm(los)
    #     if los_norm <= 0:
    #         raise Exception('invalid los input!')
    #
    #     # ------------- rotate line of sight to camera coordinates -------------
    #     R = self.camera.T_cam_to_body[:3,:3]
    #     los_cam = np.matmul(R.T, los).flatten()
    #     los_cam = los_cam / np.linalg.norm(los_cam)
    #     # TODO: we use the approximation that target range is much larger than camera-body position offset! We can do better if we calculate cone points in body frame, and then project them to the image. What range should we use in this case?
    #
    #
    #     if verbose:
    #         print('find los vectors')
    #
    #     # ------------- find cone vectors -------------
    #     # Choose a random vector w that is not parallel to los
    #     w = np.array([1, 0, 0]) if np.dot(los_cam, [1, 0, 0]) <= 0.707 else np.array([0, 1, 0])
    #     # Project w onto the plane perpendicular to v
    #     w_proj = w - np.dot(w, los_cam) * los_cam
    #     # Normalize the projection of w
    #     w_proj = w_proj / np.linalg.norm(w_proj)
    #     # Now we have a vector perpendicular to v (w_proj)
    #     # The desired vector u is a combination of v and w_proj such that the angle between them is theta
    #     u = np.cos(los_angular_uncertainty) * los_cam + np.sin(los_angular_uncertainty) * w_proj
    #
    #     # to find multiple vectors with the same angle theta from los, rotate u around the los
    #     n = num_points
    #     a = np.linspace(0, (2 * np.pi) * ((n - 1) / n), n, dtype=np.float32)
    #     ui = np.zeros((n, 3), dtype=np.float32)
    #     for i in range(n):
    #         rot_vec = los_cam * a[i]
    #         R, _ = cv2.Rodrigues(rot_vec)
    #         ui[i, :] = np.matmul(R, u)
    #         # print('ui angle to los: {}'.format(np.arccos(np.dot(los, ui[i, :]))*180/np.pi))
    #
    #     # ------------- project cone points to camera image -------------
    #     # handle special hard cases
    #     los_is_close_to_z = los_cam[2] > los_cam[0] * 1e6 and los_cam[2] > los_cam[1] * 1e6
    #     if los_angular_uncertainty > np.pi * 0.9:
    #         # in this case los uncertainty is almost 360 deg, so roi is the full image
    #         if verbose:
    #             print('special case - los uncertainty is more than 162deg. ROI is full image!')
    #         # make sure ROi is full
    #         image_points = np.array(((-1, -1),
    #                                  (self.camera.image_size[0] + 1, -1),
    #                                  (self.camera.image_size[0] + 1, self.camera.image_size[1] + 1),
    #                                  (-1, self.camera.image_size[1] + 1)), dtype=np.float32)
    #
    #     elif los_is_close_to_z and np.abs(los_angular_uncertainty - np.pi/2) < 1e-6:
    #         if los_cam[2] > 0:
    #             # in this case all cone points will have invalid projection to camera image!
    #             if verbose:
    #                 print('special case - los is camera z and los uncertainty is 90deg! ROI is full image!')
    #             # make sure ROi is full
    #             image_points = np.array(((-1, -1),
    #                                       (self.camera.image_size[0]+1, -1),
    #                                       (self.camera.image_size[0]+1, self.camera.image_size[1]+1),
    #                                       (-1, self.camera.image_size[1]+1)), dtype=np.float32)
    #         if los_cam[2] < 0:
    #             # in this case all cone points will have invalid projection to camera image!
    #             if verbose:
    #                 print('special case - los is camera -z and los uncertainty is 90deg! ROI is empty!')
    #             # make sure ROI is empty
    #             image_points = np.array(((-10, -1),
    #                                       (-10, -10),
    #                                       (-1, -10),
    #                                       (-1, -1)), dtype=np.float32)
    #
    #     elif los_cam[ 2] > 0 and np.all( ui[:, 2] < 1e-6):
    #         # in this case all cone points are behind camera, but the ROI is the full image!
    #         if verbose:
    #             print('special case - los is FW and all cone points are back')
    #         # make sure ROi is full
    #         image_points = np.array(((-1, -1),
    #                                   (self.camera.image_size[0]+1, -1),
    #                                   (self.camera.image_size[0]+1, self.camera.image_size[1]+1),
    #                                   (-1, self.camera.image_size[1]+1)), dtype=np.float32)
    #
    #     elif los_cam[ 2] < 0 and np.all( ui[:, 2] > -1e-6):
    #         """
    #         in this case los is behind the camera, but los uncertainty is very large.
    #         The correct handling of this case is complicated!
    #            - if all cone points are outside the image, ROI should be empty
    #            - if all cone points are inside the image, the correct ROI is everything outside the projected polygon
    #            - if some cone points are inside the image, the correct ROI might be made of patches
    #         For now, we set ROI to be the full image
    #         """
    #         # TODO: implement this better!
    #         if verbose:
    #             print('special case - los is back and all cone points are FW!')
    #         # make sure ROi is full
    #         image_points = np.array(((-1, -1),
    #                                   (self.camera.image_size[0]+1, -1),
    #                                   (self.camera.image_size[0]+1, self.camera.image_size[1]+1),
    #                                   (-1, self.camera.image_size[1]+1)), dtype=np.float32)
    #
    #     # handle a normal case
    #     else:
    #
    #         if verbose:
    #             print('project to image')
    #
    #         # project to camera image
    #         body_pose = g3d.rigid3dtform.Rigid3dTform(np.eye(3), (0, 0, 0))
    #         image_points, is_in_image = self.camera.project_points(ui, body_pose)
    #
    #         # handle inf cases - when a point is on the camera horizon (z=0), the projection is in infinity.
    #         is_projection_inf = np.isinf(image_points).any(axis=1)  # los perpendicular to camera Z gives inf projection
    #         if is_projection_inf.any():
    #             ui[is_projection_inf, 2] = 1e-5
    #             image_points, is_in_image = self.camera.project_points(ui, body_pose)
    #
    #         # TODO: The projection process is no good. It works strangely inm cases where angle uncertainty is large (80-90 deg)
    #         # example - check if the ROI is full ot empty using the original los, and not only the cone.
    #         # proper process:
    #         # - test if original LOS is in front or behind the camera
    #         # - test which cone points are behind the camera
    #         #
    #         #  if los is behind the camera:
    #         #  - if all cone points are behind, ROI is empty.
    #         #  - if all cone points are in front - cant be! angle uncertainty must be > 90 deg!!!
    #         #  - if some cone points are behind, and some in front: ???
    #         #         project point behind to the camera xy plane, and normalize direction vector to 1
    #
    #         #  if los is in front of the camera:
    #         #  - if all cone points are behind the camera - cant be! angle uncertainty must be > 90 deg!!!
    #         #  - if all cone points are in front the camera - do the intersection.
    #         #  - if some cone points are behind, and some in front: ???
    #         #         project point in front to the camera xy plane, and normalize direction vector to 1
    #
    #         is_in_front = ui[:,2] > 1e-9  # in front of the camera
    #         is_projection_valid = np.bitwise_not(np.isnan(image_points).any(axis=1))  # otherwise invalid projection
    #         valid_image_points = np.bitwise_and(is_in_front, is_projection_valid)
    #
    #         image_points = image_points[valid_image_points, :]
    #         # is_in_image = is_in_image[is_in_front]
    #
    #         # we can't intersect a polygon of less than 3 points!
    #         # TODO: try to find a solution to this case - maybe project the rest of the points to the plane perpendicular to camera z...
    #         if 0 < image_points.shape[0] < 3:
    #             image_points = np.zeros((0,2))
    #
    #         # if not np.any(is_in_image):  # this is a bug if polygon is bigger than the image
    #         #     valid_polygon_points = np.zeros((0, 2))
    #         #     return valid_polygon_points
    #
    #     if verbose:
    #         print('projected to image')
    #         print('intersect with image borders')
    #     image_borders = np.array(((0, 0),
    #                               (self.camera.image_size[0], 0),
    #                               (self.camera.image_size[0], self.camera.image_size[1]),
    #                               (0, self.camera.image_size[1])), dtype=np.float32)
    #     try:
    #         valid_polygon_points = g2d.polygon_2D.polygon_intersect(image_points, image_borders, use_shapely=True)
    #     except:
    #         # raise Exception('*** polygon_intersect failed! los=({},{},{}), +-{}. projected points:{} '.format(los[0], los[1], los[2], los_angular_uncertainty, image_points))
    #         valid_polygon_points = np.zeros((0, 2))
    #     if valid_polygon_points.shape[0] == image_points.shape[0] + 1:
    #         valid_polygon_points = valid_polygon_points[:-1,:2]  # remove last point. It is a duplicate of the first because shapely returns close polygons.
    #
    #     if verbose:
    #         print('intersected with image borders')
    #         print('check validity')
    #
    #     if valid_polygon_points.shape[0] < 3:
    #         if verbose:
    #             print('outside of image borders or <3 points - return empty')
    #         valid_polygon_points = np.zeros((0, 2))
    #
    #     else:
    #         if keep_num_points:
    #             if verbose:
    #                 print('adjust number of polygon points {}-{}'.format(valid_polygon_points.shape[0], num_points))
    #             valid_polygon_points = polygon_adjust_number_of_points(valid_polygon_points, num_points)
    #
    #     if int_polygon_coordinates:
    #         if verbose:
    #             print('convert to int')
    #         valid_polygon_points = np.round(valid_polygon_points).astype(np.int32)
    #
    #     return valid_polygon_points

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

    def image_bbox_to_3D_los_cov(self, bbox):
        """
        convert image bbox in pixels to 3D los and uncertainty in body frame

        Specific geometric meaning:
        los - 3D line of sight vector (x, y, z) in body frame normalized to length=1
        cov - 3x3 3D covariance matrix.
              this is calcultaed as follows:
              v1 - 3D vector = the difference between bbox center top los to bbox center los (where z=1)
              v2 - 3D vector = the difference between bbox center right los to bbox center los (where z=1)
              refer to v1, v2 as the semi major axes of the covariance matrix. Take third semi major axis as 0)
              calculate covariance matrix from these semi major axis vectors.

        :param bbox - (xtl, ytl, w, h)

        :return cov_matrix - 3x3 covariance matrix
        """

        # calc semiaxis vectors in body frame
        (xtl, ytl, w, h) = bbox
        xc = np.round(float(xtl) + float(w) / 2)
        yc = np.round(float(ytl) + float(h) / 2)
        image_points = np.array([(xc, yc),        # bbox center
                                 (xc, ytl),       # bbox up
                                 (xtl + w, yc)])  # bbox right

        R = self.camera.T_cam_to_body[:3,:3]
        los, is_in_image = self.camera.pixel_to_los(image_points, R)
        # los, is_in_image= self.pixel_to_los(image_points)

        # normalize central los to be size 1
        # normalize all other los vectors so that their projection on the central los is 1
        los[0, :] = los[0, :] / np.linalg.norm(los[0, :])
        los[1, :] = los[1, :] / np.dot(los[0, :], los[1, :])
        los[2, :] = los[2, :] / np.dot(los[0, :], los[2, :])

        # print('test los normalization:')
        # print('los norm {}'.format(np.linalg.norm(los[0, :])))
        # print('los 1 projection {}'.format(np.dot(los[0, :], los[1, :])))
        # print('los 2 projection {}'.format(np.dot(los[0, :], los[2, :])))

        # calc bbox up and right vectors
        v1 = los[1, :] - los[0, :]
        v2 = los[2, :] - los[0, :]
        v3 = los[0, :]

        # make sure the vectors are perpendicular
        # there is a small approximation here if v1, v2, v3 are not perfectly perpendicular
        s1 = np.linalg.norm(v1)
        n1 = v1 / s1
        s2 = np.linalg.norm(v2)
        n2 = v2 / s2
        n3 = v3 / np.linalg.norm(v3)

        n2_p = np.cross(n3, n1)
        v2_p = n2_p * s2  # bbox right vector (perpendicular part)
        v1_p = np.cross(n2_p, n3) * s1   # bbox up vector (perpendicular part)
        v3 = np.array((0, 0, 0))   # los vector size 0 - we have no depth perception

        # calc covariance matrix
        cov_matrix = cov_from_semiaxes(v1_p, v2_p, v3)
        central_los = los[0, :]

        return {'los': central_los, 'cov': cov_matrix}


    def save_camera(self, output_camera_params_file):
        """
        save pinhole camera parameters
        param: output_camera_params_file - output file path
        """
        if self.camera is None:
            raise Exception('camera not set!')
        self.camera.save(output_camera_params_file)
        return

    def print_camera(self):
        """
        print pinhole camera parameters
        """

        print('LosPixelConverter - camera params')
        print('   image size ({},{})   '.format(self.camera.image_size[0],self.camera.image_size[1]))
        print('   focal length ({},{})   '.format(self.camera.focal_length[0], self.camera.focal_length[1]))
        print('   principal point ({},{})   '.format(self.camera.principal_point[0], self.camera.principal_point[1]))
        print('   skew {}   '.format(self.camera.skew))
        if self.camera.T_cam_to_body is None:
            print('   T_cam_to_body is None')
        else:
            print('   T_cam_to_body ({}, {}, {})   '.format(self.camera.T_cam_to_body[0, 0], self.camera.T_cam_to_body[0, 1], self.camera.T_cam_to_body[0, 2]))
            print('                 ({}, {}, {})   '.format(self.camera.T_cam_to_body[1, 0], self.camera.T_cam_to_body[1, 1], self.camera.T_cam_to_body[1, 2]))
            print('                 ({}, {}, {})   '.format(self.camera.T_cam_to_body[2, 0], self.camera.T_cam_to_body[2, 1], self.camera.T_cam_to_body[2, 2]))

        return


def polygon_adjust_number_of_points(polygon_points, required_num_points, reduction_method='uniform'):
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

        method 1: iteratively remove the smallest 3-vertex triangle.
        method 2: uniform sample

    param: polygon_points - polygon points (nx2). must be ordered clockwise or anti-clockwise
    param: required_num_points - number of required points
    """

    polygon_points = np.array(polygon_points)
    if polygon_points.shape[1] != 2:
        raise Exception('invalid polygon shape. must be (nx2)!')
    if required_num_points < 3:
        raise Exception('invalid number of required points. must be >= 3!')

    n = polygon_points.shape[0]

    # reduce collinear vertices (only if we need to reduce)
    if n > required_num_points:
        polygon_points = remove_collinear_vertices(polygon_points)
        n = polygon_points.shape[0]

    if required_num_points == n:
        res_polygon = polygon_points  # don't need to change anything

    elif required_num_points > n:
        res_polygon = add_collinear_vertices(polygon_points, required_num_points)

    else:  # required_num_points < n
        n_remove = n - required_num_points
        res_polygon = simplify_polygon_uniform_angular_sampling(polygon_points, required_num_points)
        # res_polygon = simplify_polygon_max_area_greedy(polygon_points, required_num_points)

    return res_polygon

def simplify_polygon_uniform_angular_sampling(points, m):
    """
    reduce polygon to nm points by uniform angular sampling
    :param points: nx2 numpy array
    :param m: number of required points
    :return:
    """

    # ensure numpy array and closed polygon
    pts = np.asarray(points)
    if not np.all(pts[0] == pts[-1]):
        pts = np.vstack([pts, pts[0]])
    if pts.shape[0] < m:
        raise Exception('polygon has less points than required')

    # compute cumulative perimeter distances
    dists = np.sqrt(((np.diff(pts, axis=0))**2).sum(axis=1))
    cumdist = np.insert(np.cumsum(dists), 0, 0)
    total = cumdist[-1]

    # sample m evenly spaced distances
    target = np.linspace(0, total, m+1)[:-1]  # m points, exclude last duplicate

    # interpolate along edges
    simplified = []
    for t in target:
        i = np.searchsorted(cumdist, t) - 1
        i = np.clip(i, 0, len(pts)-2)
        ratio = (t - cumdist[i]) / dists[i]
        p = pts[i] + ratio * (pts[i+1] - pts[i])
        simplified.append(p)

    return np.array(simplified)

def simplify_polygon_max_area_greedy(points, m):
    """
    reduce polygon to m points by removing least area vertex in a greedy manner
    :param points: nx2 numpy array
    :param m: number of required points
    :return:
    """

    # ensure numpy array and closed polygon
    res_polygon = np.asarray(points)
    if not np.all(res_polygon[0] == res_polygon[-1]):
        res_polygon = np.vstack([res_polygon, res_polygon[0]])

    n = res_polygon.shape[0]
    n_remove = n - m
    if m < 0:
        raise Exception('polygon has less points than required')

    # calc triangle area reduction for each vertex
    tri_area = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = np.array(res_polygon[(i - 1) % n])
        b = np.array(res_polygon[i % n])
        c = np.array(res_polygon[(i + 1) % n])
        tri_area[i] = triangle_area(a, b, c)

    # iteratively remove smallest triangle
    for i in range(n_remove):
        # find smallest area triangle
        idx = np.argmin(tri_area)

        # fix neighboring triangle area calculation
        m = res_polygon.shape[0]
        a1 = np.array(res_polygon[(idx - 1) % m])
        b1 = np.array(res_polygon[(idx + 1) % m])
        c1 = np.array(res_polygon[(idx + 2) % m])
        tri_area[(i + 1) % m] = triangle_area(a1, b1, c1)

        a2 = np.array(res_polygon[(idx - 2) % m])
        b2 = np.array(res_polygon[(idx - 1) % m])
        c2 = np.array(res_polygon[(idx + 1) % m])
        tri_area[(i - 1) % m] = triangle_area(a2, b2, c2)

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


def remove_collinear_vertices(points):
    """
    reduce strait line vertixes from polygons
    :param points: nx2 numpy array
    :param m: number of required points
    :return:
    """

    pts = np.asarray(points, float)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    n = len(pts)

    # calculate triangle area for each vertex
    prev = np.roll(pts, 1, axis=0)
    next_ = np.roll(pts, -1, axis=0)
    triangle_area = np.abs((next_[:,0]-pts[:,0])*(prev[:,1]-pts[:,1]) -
                   (next_[:,1]-pts[:,1])*(prev[:,0]-pts[:,0]))

    # calculate triangle area threshold
    triangle_area_th = 1e-3 * np.median(triangle_area)

    # remove vertexes with small triangle area
    keep = triangle_area > triangle_area_th
    keep[0] = keep[-1] = True  # always keep first/last

    cleaned = pts[keep]
    if not np.allclose(cleaned[0], cleaned[-1]):
        cleaned = np.vstack([cleaned, cleaned[0]])

    return cleaned


def add_collinear_vertices(points, m):
    """
    add vertexs on polygon edges to get m vertexs in total
    :param points: nx2 numpy array
    :param m: number of required points
    :return:
    """

    # add vertexes on long edges
    points = np.asarray(points)
    n = points.shape[0]
    n_missing = m - n
    if n_missing < 0:
        raise Exception('polygon too large! polygon has {} pointes, and cannot be increased to get the {} required'.format(n, m))

    res_polygon = points
    shifted = np.roll(points, -1, axis=0)
    edge_sizes = np.linalg.norm(points - shifted, axis=1)
    for i in range(n_missing):
        # find the longest edge
        idx = np.argmax(edge_sizes)
        # add a point in the middle of the longest edge
        n = res_polygon.shape[0]
        p = (res_polygon[idx, :] + res_polygon[(idx + 1) % n, :]) / 2
        res_polygon = np.insert(res_polygon, (idx + 1) % n, p, axis=0)
        # fix new edge sizes
        d = edge_sizes[idx]/2
        edge_sizes[idx] = d/2
        edge_sizes = np.insert(edge_sizes, (idx + 1) % n, d, axis=0)

    return res_polygon

def calc_target_los_angular_uncertainty(prior_angular_uncertainty=0, relative_position_uncertainty=None,
                                        range_to_target=None, angular_uncertainty_limit=np.pi):
    """
    calc angular uncertainty cone for some target
    The assumption is that the target is inside an uncertainty cone with a specific angular radius
    to calculate the estimated cone is determined by two factors:
    1. prior angular uncertainty in los to the target
    2. prior target position uncertainty
    This function merges both these parameters into one uncertainty cone estimation.

    :param prior_angular_uncertainty - prior angular uncertainty in [rad]
                                       target los uncertainty cone angular radius
    :param relative_position_uncertainty - relative position uncertainty in [m]
                                           this means the target might be anywhere in this radius from the estimated position
    :param range_to_target - range between us and the target in [m]
    :param angular_uncertainty_limit - top limit for total uncertainty cone angular radius [rad].
                                       default is 180 degrees (which make the uncertainty cone become a full sphere)
                                       in many cases it makes sense to limit to 90 degrees (which make the uncertainty cone become a half sphere)

    :return angular_cone_radius - this is the estimated raduis of uncertainty cone
    """

    if angular_uncertainty_limit > np.pi:
        raise Exception('angular_uncertainty_limit must be < pi (180deg)')

    if relative_position_uncertainty is None and range_to_target is None:
        position_induced_uncertainty = 0
    elif relative_position_uncertainty is not None and range_to_target is not None:
        if relative_position_uncertainty > range_to_target:
            position_induced_uncertainty = np.pi  # target may be in any direction
        else:
            position_induced_uncertainty = np.arctan2(np.abs(relative_position_uncertainty), range_to_target)
    else:
        raise Exception('invalid input! relative_position_uncertainty and range_to_target must be set together!')

    angular_cone_radius = min(float(prior_angular_uncertainty) + float(position_induced_uncertainty), angular_uncertainty_limit)  # can't be more than 180 degrees

    return angular_cone_radius



def cov_from_semiaxes(a1, a2, a3, return_eigs=False):
    """
    Compute 3×3 covariance matrix from three semimajor axis vectors (each = direction * std).
    The norm of ai is the standard deviation along that direction.

    Parameters
    ----------
    a1, a2, a3 : array-like, shape (3,)
        Semi-major axis vectors.
    return_eigs : bool, optional
        If True, also return eigenvalues, eigenvectors, and std deviations.

    Returns
    -------
    Sigma : ndarray, shape (3, 3)
        Covariance matrix.
    eigvals : ndarray, shape (3,), optional
        Eigenvalues (variances, sorted descending).
    eigvecs : ndarray, shape (3, 3), optional
        Corresponding orthonormal eigenvectors (columns).
    stds : ndarray, shape (3,), optional
        Standard deviations along the principal axes = sqrt(eigvals).
    """
    A = [np.asarray(a, float) for a in (a1, a2, a3)]
    Sigma = np.zeros((3, 3))
    for a in A:
        Sigma += np.outer(a, a)
    Sigma = 0.5 * (Sigma + Sigma.T)  # ensure symmetry

    if not return_eigs:
        return Sigma

    # eigen-decomposition (ascending order → reverse to descending)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    stds = np.sqrt(np.maximum(eigvals, 0))  # guard against tiny negative due to rounding

    return Sigma, eigvals, eigvecs, stds




def closest_distance_between_trajectories(traj1, traj2):
    """
    Compute the minimum distance between two piecewise-linear trajectories
    parameterized by time.

    Each trajectory is a list of (t, x, y, [z]) points.
    Motion between points is linear interpolation.

    Returns:
        t_min : float - global time of closest approach
        d_min : float - minimal distance
        p1_min, p2_min : np.ndarray - positions of each trajectory at that time
    """

    traj1 = np.array(traj1, dtype=float)
    traj2 = np.array(traj2, dtype=float)

    t1, p1 = traj1[:, 0], traj1[:, 1:]
    t2, p2 = traj2[:, 0], traj2[:, 1:]

    d_min = np.inf
    t_min = None
    p1_min = None
    p2_min = None

    t2s = t2[:-1]
    t2f = t2[1:]
    for i in range(len(t1) - 1):
        idx2 = t2s <= t1[i] and t2s <= t1[i]
        for j in range(len(t2) - 1):

            # Overlap in time
            t_start = max(t1[i], t2[j])
            t_end   = min(t1[i+1], t2[j+1])
            if t_end <= t_start:
                continue  # segments don't overlap in time

            # Segment definitions
            v1 = (p1[i+1] - p1[i]) / (t1[i+1] - t1[i])
            v2 = (p2[j+1] - p2[j]) / (t2[j+1] - t2[j])

            # Positions at t_start
            p1s = p1[i] + v1 * (t_start - t1[i])
            p2s = p2[j] + v2 * (t_start - t2[j])

            # Relative motion: p_rel(t) = (p1s - p2s) + (v1 - v2)*(t - t_start)
            dp0 = p1s - p2s
            dv = v1 - v2

            # Minimize |dp0 + dv*(t - t_start)|^2 over [t_start, t_end]
            a = np.dot(dv, dv)
            b = np.dot(dp0, dv)

            if a == 0:  # no relative velocity, constant distance
                t_candidate = t_start
            else:
                t_candidate = t_start - b / a
                # clamp to valid time range
                t_candidate = np.clip(t_candidate, t_start, t_end)

            # Evaluate positions at this time
            p1_c = p1[i] + v1 * (t_candidate - t1[i])
            p2_c = p2[j] + v2 * (t_candidate - t2[j])
            d = np.linalg.norm(p1_c - p2_c)

            if d < d_min:
                d_min = d
                t_min = t_candidate
                p1_min = p1_c
                p2_min = p2_c

    return t_min, d_min, p1_min, p2_min



def random_vector_with_angle_std(v, theta_std):
    """
    Generate a random 3D vector at a an angle `theta` (in radians) to vector `v`,
    where 'theta' is sampled from a normal distribution=N(0, theta_std)

    Parameters:
    - v: (3x1) vector
    - theta_std: float → std for angle in radians

    Returns:
    - u: np.array of shape (3,) → random vector at angle `theta` to v
    """
    theta = np.random.normal(0.0, theta_std)
    return _random_vector_at_angle(v, theta)


def _random_vector_at_angle(v, theta):
    """
    Generate a random 3D vector at a fixed angle `theta` (in radians) to vector `v`.

    Parameters:
    - v: (3x1) vector
    - theta: float → desired angle in radians

    Returns:
    - u: np.array of shape (3,) → random vector at angle `theta` to v
    """
    v = np.asarray(v, dtype=float)
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        raise ValueError("Input vector v must be non-zero.")
    v_hat = v / v_norm

    # Step 1: Find a vector perpendicular to v
    if np.allclose(v_hat, [1, 0, 0]):
        # If v is along x-axis, pick y-axis for cross
        va = np.cross(v_hat, [0, 1, 0])
    else:
        va = np.cross(v_hat, [1, 0, 0])
    va /= np.linalg.norm(va)

    # Step 2: Find another perpendicular vector
    vb = np.cross(v_hat, va)

    # Step 3: Random rotation around v
    phi = np.random.uniform(0, 2 * np.pi)
    u = np.cos(theta) * v_hat + np.sin(theta) * (np.cos(phi) * va + np.sin(phi) * vb)

    return u

if __name__ == '__main__':

    # print('calc_target_los_angular_uncertainty test:')
    # prior_angular_uncertainty = np.pi/9
    # relative_position_uncertainty = 10
    # for i in range(50,-50,-1):
    #     angular_uncertainty = calc_target_los_angular_uncertainty(prior_angular_uncertainty=prior_angular_uncertainty,
    #                             relative_position_uncertainty=relative_position_uncertainty,
    #                                                               range_to_target=np.abs(i), angular_uncertainty_limit=np.pi)
    #     print('range: {} - {}'.format(i, angular_uncertainty*180/np.pi))

    image_file = '/home/roee/Projects/datasets/interceptor_drone/20250701_kfar_galim/2025-07-01_09-35-10/camera_2025_7_1-6_35_13_extracted/images/000.png'
    camera_calibration_file = '/home/roee/Projects/datasets/interceptor_drone/20250612_calibration/20250612_pz001_calibration/camera_intrinsics_IDC1.yaml'

    test_pattern = 'yaw'  # 'yaw'/ 'pitch' / 'random'

    if test_pattern == 'yaw':
        # generate los in a circle
        n = 200
        a = np.linspace(0, np.pi*2, n)
        los_x = np.cos(a)
        los_z = np.zeros_like(a)
        los_y = np.sin(a)
        los_angular_uncertainty = 180*np.pi/180 * np.ones(n)  # 20*np.pi/180

    if test_pattern == 'pitch':
        # generate los in a circle
        n = 200
        a = np.linspace(0, np.pi*2, n)
        los_z = np.cos(a)
        los_y = np.zeros_like(a)
        los_x = np.sin(a)
        los_angular_uncertainty = 40*np.pi/180 * np.ones(n)  # 20*np.pi/180

    elif  test_pattern == 'random':
        # generate random los
        np.random.seed(7)
        n=10000
        los_x = np.random.rand(n) * 2 - 1
        los_y = np.random.rand(n) * 2 - 1
        los_z = np.random.rand(n) * 2 - 1
        a = np.linspace(0,n-1, n) * np.pi/180  # just plot frame id
        los_angular_uncertainty = np.random.rand(n) * (np.pi/2) + 0.1 # 5-90 deg

    los = np.vstack((los_x, los_y, los_z)).T
    los = los / np.sqrt(np.sum(np.power(los, 2), axis=1)).reshape((n,1))


    # load image
    img = cv2.imread(image_file)

    # load camera
    cam = cv_core.pinhole_camera.PinholeCamera()

    # cam.load(camera_calibration_file)
    image_size = [640, 480]
    intrinsic_matrix = np.array([[580, 0, 320],
                                 [0, 580, 240],
                                 [0, 0, 1]])
    dist_coeffs = np.array([0,0,0,0,0])
    T_c2b = np.array([[0.0, 0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
    cam.set('cam0', cv_core.pinhole_camera.CameraModel.PINHOLE, intrinsic_matrix, dist_coeffs, image_size, T_cam_to_body=T_c2b)

    camera_intrinsic_matrix = cam.K
    distortion_coefficients = cam.distortion_coefficients
    image_size = cam.image_size
    # camera_extrinsic_matrix = np.eye(4)  # cam.T_cam_to_body
    camera_extrinsic_matrix = cam.T_cam_to_body

    lpc =  LosPixelConverter()
    print('set camera test:')
    lpc.set_camera(camera_intrinsic_matrix, distortion_coefficients, image_size, camera_extrinsic_matrix)
    lpc.print_camera()
    print('load camera test:')
    lpc.load_camera(camera_calibration_file)
    lpc.print_camera()

    lpc.set_camera(camera_intrinsic_matrix, distortion_coefficients, image_size, camera_extrinsic_matrix)


    for i in range(n):
        img_draw = copy.deepcopy(img)
        roi_polygon =  lpc.image_polygon_from_los(los[i, :], los_angular_uncertainty[i], num_points=12,
                                                  int_polygon_coordinates=False, keep_num_points=False, verbose=True)

        roi_polygon = np.round(roi_polygon).astype(np.int32)
        points = roi_polygon.reshape((-1, 1, 2))

        cv2.polylines(img_draw, [points], isClosed=True, color=(0, 255, 0), thickness=3)
        img_draw = cv2.putText(img_draw, 'ang={:.2f}'.format(a[i]*180/np.pi), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (50, 255, 50), 1, lineType=cv2.LINE_AA)
        cv2.imshow('image with roi polygon', img_draw)
        cv2.waitKey(50)
        # print('ang={:.2f}'.format(a[i]*180/np.pi))
        # print(roi_polygon)
        aa=5

    print('done!')
    cv2.destroyAllWindows()