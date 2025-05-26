import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2

def plot_3d_cov_ellipsoid(mean, cov, ax=None, n_std=2, color='skyblue'):

    num_points = 20

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Get the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvectors by eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Generate data for the ellipsoid
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    xyz = np.stack((x, y, z), axis=-1)

    # Scale ellipsoid axes by the sqrt of eigenvalues and desired n_std
    chi2_val = np.sqrt(chi2.ppf(0.95, df=3))  # 95% confidence interval
    axes_lengths = n_std * np.sqrt(eigenvalues)

    for i in range(xyz.shape[0]):
        for j in range(xyz.shape[1]):
            xyz[i, j] = mean + eigenvectors @ (xyz[i, j] * axes_lengths)

    # Plot
    ax.plot_surface(
        xyz[:, :, 0],
        xyz[:, :, 1],
        xyz[:, :, 2],
        color=color,
        alpha=0.5,
        edgecolor='none'
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax


if __name__ == '__main__':
    # Example usage:
    mean = np.array([0, 0, 0])
    cov = np.array([
        [4, 1, 1],
        [1, 3, 0.5],
        [1, 0.5, 2]
    ])

    plot_3d_cov_ellipsoid(mean, cov)
    plt.show()