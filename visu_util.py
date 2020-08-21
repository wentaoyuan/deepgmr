'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_pcd(ax, pcd, color=None, cmap='viridis', size=4, alpha=0.9, azim=60, elev=0):
    if color is None:
        color = pcd[:, 0]
        vmin = -2
        vmax = 1.5
    else:
        vmin = 0
        vmax = 1
    ax.view_init(azim=azim, elev=elev)
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color, s=size, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    min_lim = min(pcd.min() * 0.9, lims.min())
    max_lim = max(pcd.max() * 0.9, lims.max())
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((min_lim, max_lim))
    ax.set_axis_off()


def plot_gmm(ax, mix, mu, cov, color=None, cmap='viridis', azim=60, elev=0, numWires=15, wireframe=True):
    if color is None:
        color = np.arange(mix.shape[0]) / (mix.shape[0] - 1)
    if cmap is not None:
        cmap = cm.get_cmap(cmap)
        color = cmap(color)

    u = np.linspace(0.0, 2.0 * np.pi, numWires)
    v = np.linspace(0.0, np.pi, numWires)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v)) 
    XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])

    alpha = mix / mix.max()
    ax.view_init(azim=azim, elev=elev)

    for k in range(mix.shape[0]):
        # find the rotation matrix and radii of the axes
        U, s, V = np.linalg.svd(cov[k])
        x, y, z = V.T @ (np.sqrt(s)[:, None] * XYZ) + mu[k][:, None]
        x = x.reshape(numWires, numWires)
        y = y.reshape(numWires, numWires)
        z = z.reshape(numWires, numWires)
        if wireframe:
            ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color=color[k], alpha=alpha[k])
        else:
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color[k], alpha=alpha[k])


def visualize(inputs):
    for i in range(len(inputs)):
        inputs[i] = inputs[i].detach().cpu().numpy()
    p1, gamma1, pi1, mu1, sigma1, p2, gamma2, pi2, mu2, sigma2, \
        p1_trans, init_r_err, init_t_err, init_rmse, r_err, t_err, rmse = inputs

    fig = plt.figure(figsize=(8, 8))
    title = 'Rotation error {:.2f}\nTranslation error {:.4f}\nRMSE {:.4f}'

    ax = fig.add_subplot(221, projection='3d')
    plot_pcd(ax, p1, cmap='Reds')
    plot_pcd(ax, p2, cmap='Blues')
    ax.set_title(title.format(init_r_err, init_t_err, init_rmse))

    ax = fig.add_subplot(222, projection='3d')
    plot_pcd(ax, p1_trans, cmap='Reds')
    plot_pcd(ax, p2, cmap='Blues')
    ax.set_title(title.format(r_err, t_err, rmse))

    ax = fig.add_subplot(223, projection='3d')
    color1 = np.argmax(gamma1, axis=1) / (gamma1.shape[1] - 1)
    plot_pcd(ax, p1, color1)
    plot_gmm(ax, pi1, mu1, sigma1)
    ax.set_title('Source GMM')

    ax = fig.add_subplot(224, projection='3d')
    color2 = np.argmax(gamma2, axis=1) / (gamma2.shape[1] - 1)
    plot_pcd(ax, p2, color2)
    plot_gmm(ax, pi2, mu2, sigma2)
    ax.set_title('Target GMM')

    plt.tight_layout()
    return fig
