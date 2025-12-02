""" 3D geometry tools
"""
import numpy as np
import geometry_utils
import cv_core
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def closest_point_piecewise_linear_trajectories(verbose=True, draw=False):
    points1 = np.array([[0, 0, 0],
                      [1, 1, 0],
                      [2, 2, 1],
                      [3, 2, 2],
                      [3, 2, 2],
                      [4, 2, 3],
                      [5, 3, 4],
                      [6, 4, 4]])

    points2 = np.array([[0, 0, 10],
                      [1, 1, 10],
                      [2, 2, 7],
                      [3, 2, 6],
                      [4, 2, 7],
                      [5, 3, 10],
                      [6, 4, 10]])

    t1 = np.array([0,1,2,2.5,3.5,4,5,6])
    t2 = np.array(range(7))

    t_min, d_min, p1_min, p2_min = geometry_utils.closest_point_piecewise_linear_trajectories(points1, t1, points2, t2)
    if verbose:
        print('trajectory interception:')
        print('min dist = {}'.format(d_min))
        print('t = {}'.format(t_min))
        print('pos1 = {}'.format(p1_min))
        print('pos2 = {}'.format(p2_min))

    ret = t_min==3 and d_min == 4 and all(p1_min==[3, 2, 2]) and all(p2_min==[3, 2, 6])
    return ret


def lospixelconverter_image_bbox_to_3D_los_cov_test(verbose=True, draw=False):

    # bbox = [100, 200, 15, 18]
    bbox = [300, 210, 40, 60]  # 40x60 around the center

    # set LosPixelConverter with camera
    cam = cv_core.pinhole_camera.PinholeCamera()
    image_size = [640, 480]
    intrinsic_matrix = np.array([[580, 0, 320],
                                 [0, 580, 240],
                                 [0, 0, 1]])
    dist_coeffs = np.array([0,0,0,0,0])
    T_c2b = np.array([[0.0, 0.5,                 0.8660254037844386, 0.0],    # body FRD, camera fw and 30 deg up
                      [1.0, 0.0,                 0.0,                0.0],
                      [0.0, 0.8660254037844386, -0.5,                0.0],
                      [0.0, 0.0,                 0.0,                1.0]])
    # T_c2b = np.array([[0.0, 0.0, 1.0, 0.0],  # body FRD, camera fw
    #                   [1.0, 0.0, 0.0, 0.0],
    #                   [0.0, 1.0, 0.0, 0.0],
    #                   [0.0, 0.0, 0.0, 1.0]])
    # T_c2b = np.eye(4)  # body frame is the same as camera frame

    cam.set('cam0', cv_core.pinhole_camera.CameraModel.PINHOLE, intrinsic_matrix, dist_coeffs, image_size, T_cam_to_body=T_c2b)

    camera_intrinsic_matrix = cam.K
    distortion_coefficients = cam.distortion_coefficients
    image_size = cam.image_size
    camera_extrinsic_matrix = cam.T_cam_to_body

    if verbose:
        print('LosPixelConverter lospixelconverter_image_bbox_to_3D_los_cov_test test')
        print('set LosPixelConverter:')
    lpc =  geometry_utils.LosPixelConverter()
    lpc.set_camera(camera_intrinsic_matrix, distortion_coefficients, image_size, camera_extrinsic_matrix)
    if verbose:
        lpc.print_camera()

    res = lpc.image_bbox_to_3D_los_cov(bbox)
    if verbose:
        print('converting bbox: {}'.format(bbox))
        print('los: {}'.format(res['los']))
        print('cov: {}'.format(res['cov']))

    if draw:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot camera
        camera_position = lpc.camera.T_cam_to_body[:3, 3]
        ax.plot(camera_position[0], camera_position[1], camera_position[2], '*b')

        # plot covariance
        ax, xyz_lims = plot_cov_ellipsoid(res['cov'], mean=res['los'], n_std=1.0, ax=ax, color='c', alpha=0.3)

        # plot central los
        p1 = camera_position
        p2 = camera_position + res['los']
        ax.plot((p1[0], p2[0]), (p1[1], p2[1]), (p1[2], p2[2]), '-r')

        # handle axis
        X = (xyz_lims['x'][0], xyz_lims['x'][0], camera_position[0])
        Y = (xyz_lims['y'][0], xyz_lims['y'][0], camera_position[1] )
        Z = (xyz_lims['z'][0], xyz_lims['z'][0], camera_position[2] )
        set_axes_equal(ax, X, Y, Z, padding=0.1)

        plt.show()

    return True


def plot_cov_ellipsoid(Sigma, mean=[0, 0, 0], n_std=1.0, ax=None, color='c', alpha=0.3):
    """
    Plots a 3D covariance ellipsoid.

    Parameters:
    -----------
    Sigma : 3x3 covariance matrix
    mean  : 3-element list, center of ellipsoid
    n_std : scale factor (number of standard deviations)
    ax    : optional matplotlib 3D axis
    color : face color
    alpha : transparency
    """
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    radii = n_std * np.sqrt(eigvals)  # axes lengths

    # Create a sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Stack and scale by radii
    sphere = np.stack((x, y, z), axis=-1)  # shape (50,50,3)
    for i in range(3):
        sphere[..., i] *= radii[i]

    # Rotate sphere by eigenvectors
    ellipsoid = sphere @ eigvecs.T
    ellipsoid += mean  # translate

    # Plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2],
        rstride=2, cstride=2, color=color, alpha=alpha, edgecolor='k'
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set equal aspect
    max_radius = np.max(radii)
    # ax.set_xlim(mean[0] - max_radius, mean[0] + max_radius)
    # ax.set_ylim(mean[1] - max_radius, mean[1] + max_radius)
    # ax.set_zlim(mean[2] - max_radius, mean[2] + max_radius)

    xlims = [mean[0] - max_radius, mean[0] + max_radius]
    ylims = [mean[1] - max_radius, mean[1] + max_radius]
    zlims = [mean[2] - max_radius, mean[2] + max_radius]
    xyz_lims = {'x': xlims, 'y': ylims, 'z': zlims}

    return ax, xyz_lims

def set_axes_equal(ax, X, Y, Z, padding=0.1):
    """
    Set 3D axes to equal scale and auto-limit based on data.

    Parameters:
    - ax : matplotlib 3d axis
    - X, Y, Z : array-like, coordinates of all points to plot
    - padding : fraction of max range to pad around the data
    """
    X, Y, Z = np.asarray(X), np.asarray(Y), np.asarray(Z)
    xlim = [X.min(), X.max()]
    ylim = [Y.min(), Y.max()]
    zlim = [Z.min(), Z.max()]

    # Find max range
    max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])

    # Center each axis
    xmid = np.mean(xlim)
    ymid = np.mean(ylim)
    zmid = np.mean(zlim)

    # Set limits with optional padding
    pad = max_range * padding
    ax.set_xlim(xmid - max_range / 2 - pad, xmid + max_range / 2 + pad)
    ax.set_ylim(ymid - max_range / 2 - pad, ymid + max_range / 2 + pad)
    ax.set_zlim(zmid - max_range / 2 - pad, zmid + max_range / 2 + pad)

if __name__ == '__main__':

    # if lospixelconverter_image_bbox_to_3D_los_cov_test(verbose=True, draw=True):
    #     print("lospixelconverter_image_bbox_to_3D_los_cov_test OK!")
    # else:
    #     print("lospixelconverter_image_bbox_to_3D_los_cov_test Failed!")

    if closest_point_piecewise_linear_trajectories(verbose=True, draw=False):
        print("closest_point_piecewise_linear_trajectories OK!")
    else:
        print("closest_point_piecewise_linear_trajectories Failed!")
