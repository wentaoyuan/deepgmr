'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import h5py
import numpy as np
import os
import torch
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


def knn_idx(pts, k):
    kdt = cKDTree(pts) 
    _, idx = kdt.query(pts, k=k+1)
    return idx[:, 1:]


def get_rri(pts, k):
    # pts: N x 3, original points
    # q: N x K x 3, nearest neighbors
    q = pts[knn_idx(pts, k)]
    p = np.repeat(pts[:, None], k, axis=1)
    # rp, rq: N x K x 1, norms
    rp = np.linalg.norm(p, axis=-1, keepdims=True)
    rq = np.linalg.norm(q, axis=-1, keepdims=True)
    pn = p / rp
    qn = q / rq
    dot = np.sum(pn * qn, -1, keepdims=True)
    # theta: N x K x 1, angles
    theta = np.arccos(np.clip(dot, -1, 1))
    T_q = q - dot * p
    sin_psi = np.sum(np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
    cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
    psi = np.arctan2(sin_psi, cos_psi) % (2*np.pi)
    idx = np.argpartition(psi, 1)[:, :, 1:2]
    # phi: N x K x 1, projection angles
    phi = np.take_along_axis(psi, idx, axis=-1)
    feat = np.concatenate([rp, rq, theta, phi], axis=-1)
    return feat.reshape(-1, k * 4)


def get_rri_cuda(pts, k, npts_per_block=1):
    import pycuda.autoinit
    mod_rri = SourceModule(open('rri.cu').read() % (k, npts_per_block))
    rri_cuda = mod_rri.get_function('get_rri_feature')

    N = len(pts)
    pts_gpu = gpuarray.to_gpu(pts.astype(np.float32).ravel())
    k_idx = knn_idx(pts, k)
    k_idx_gpu = gpuarray.to_gpu(k_idx.astype(np.int32).ravel())
    feat_gpu = gpuarray.GPUArray((N * k * 4,), np.float32)

    rri_cuda(pts_gpu, np.int32(N), k_idx_gpu, feat_gpu,
             grid=(((N-1) // npts_per_block)+1, 1),
             block=(npts_per_block, k, 1))
    
    feat = feat_gpu.get().reshape(N, k * 4).astype(np.float32)
    return feat


def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)


class TestData(Dataset):
    def __init__(self, path, args):
        super(TestData, self).__init__()
        with h5py.File(path, 'r') as f:
            self.source = f['source'][...]
            self.target = f['target'][...]
            self.transform = f['transform'][...]
        self.n_points = args.n_points
        self.use_rri = args.use_rri
        self.k = args.k if self.use_rri else None
        self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri

    def __getitem__(self, index):
        pcd1 = self.source[index][:self.n_points]
        pcd2 = self.target[index][:self.n_points]
        if self.use_rri:
            pcd1 = np.concatenate([pcd1, self.get_rri(pcd1 - pcd1.mean(axis=0), self.k)], axis=1)
            pcd2 = np.concatenate([pcd2, self.get_rri(pcd2 - pcd2.mean(axis=0), self.k)], axis=1)
        transform = self.transform[index]
        return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32')

    def __len__(self):
        return self.transform.shape[0]


class TrainData(Dataset):
    def __init__(self, path, args):
        super(TrainData, self).__init__()
        with h5py.File(path, 'r') as f:
            self.points = f['points'][...]
        self.n_points = args.n_points
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans
        self.noisy = not args.clean
        self.use_rri = args.use_rri
        self.k = args.k if self.use_rri else None
        self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri

    def __getitem__(self, index):
        pcd1 = self.points[index][:self.n_points]
        pcd2 = self.points[index][:self.n_points]
        transform = random_pose(self.max_angle, self.max_trans / 2)
        pose1 = random_pose(np.pi, self.max_trans)
        pose2 = transform @ pose1
        pcd1 = pcd1 @ pose1[:3, :3].T + pose1[:3, 3]
        pcd2 = pcd2 @ pose2[:3, :3].T + pose2[:3, 3]
        if self.noisy:
            pcd1 = jitter_pcd(pcd1)
            pcd2 = jitter_pcd(pcd2)
        if self.use_rri:
            pcd1 = np.concatenate([pcd1, self.get_rri(pcd1 - pcd1.mean(axis=0), self.k)], axis=1)
            pcd2 = np.concatenate([pcd2, self.get_rri(pcd2 - pcd2.mean(axis=0), self.k)], axis=1)
        return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32')

    def __len__(self):
        return self.points.shape[0]

