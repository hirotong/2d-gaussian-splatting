#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch3d.ops import knn_points, knn_gather
from utils.image_utils import erode
from pytorch3d.loss import mesh_laplacian_smoothing


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:, 1:-1, :-2] + disp[:, 1:-1, 2:] - 2 * disp[:, 1:-1, 1:-1])
    grad_disp_y = torch.abs(disp[:, :-2, 1:-1] + disp[:, 2:, 1:-1] - 2 * disp[:, 1:-1, 1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss


def mask_entropy_loss(mask, gt):
    zero_epsilon = 1e-6
    mask = torch.clamp(mask, zero_epsilon, 1 - zero_epsilon)
    loss = -torch.mean(gt * torch.log(mask) + (1 - gt) * torch.log(1 - mask))
    return loss


def predicted_normal_loss(normal, normal_ref, alpha=None):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight * 255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32) / 255.0)
        weight = weight[None, ...].repeat(3, 1, 1)
        weight = weight.to(device)
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1, 2, 0).reshape(-1, 3)[..., 0].detach()
    n = normal_ref.permute(1, 2, 0).reshape(-1, 3).detach()
    n_pred = normal.permute(1, 2, 0).reshape(-1, 3)
    loss = (w * (1.0 - torch.sum(n * n_pred, dim=-1))).mean()

    return loss


def delta_normal_loss(delta_normal_norm, alpha=None):
    # delta_normal_norm: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight * 255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32) / 255.0)
        weight = weight[None, ...].repeat(3, 1, 1)
        weight = weight.to(device)
    else:
        weight = torch.ones_like(delta_normal_norm)

    w = weight.permute(1, 2, 0).reshape(-1, 3)[..., 0].detach()
    l = delta_normal_norm.permute(1, 2, 0).reshape(-1, 3)[..., 0]
    loss = (w * l).mean()

    return loss


def cam_depth2world_point(cam_z, pixel_idx, intrinsic, extrinsic):
    """
    cam_z: (1, N)
    pixel_idx: (1, N, 2)
    intrinsic: (3, 3)
    extrinsic: (4, 4)
    world_xyz: (1, N, 3)
    """
    valid_x = (pixel_idx[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    valid_y = (pixel_idx[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    ndc_xy = torch.stack([valid_x, valid_y], dim=-1)
    # inv_scale = torch.tensor([[W - 1, H - 1]], device=cam_z.device)
    # cam_xy = ndc_xy * inv_scale * cam_z[...,None]
    cam_xy = ndc_xy * cam_z[..., None]
    cam_xyz = torch.cat([cam_xy, cam_z[..., None]], dim=-1)
    world_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[..., 0:1])], axis=-1) @ torch.inverse(extrinsic).transpose(
        0, 1
    )
    world_xyz = world_xyz[..., :3]
    return world_xyz, cam_xyz


def normalize_adjacency_matrix(A):
    # Compute the degree matrix
    degrees = torch.sparse.sum(A, dim=1).to_dense()

    d_inv_sqrt = torch.pow(degrees, -0.5)

    A_normalized = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]

    # D_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))
    # D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

    # A_normalized = D_inv_sqrt @ A.to_dense() @ D_inv_sqrt
    return A_normalized


def laplacian_matrix(points, dist, indices):
    """_summary_

    _extended_summary_

    Args:
        point: (N, 3)
        dist: (N, K)
        indices: (N, K)
        eps: _description_
    """

    V = points.shape[0]

    row_indices = (
        (torch.arange(indices.shape[0]) * indices.shape[1]).repeat_interleave(indices.shape[1]).to(points.device)
    )
    col_indices = indices.flatten()

    weights = torch.exp(-dist.flatten())

    A = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices], dim=0).long(), weights, (V, V))
    A = 0.5 * (A + A.t())

    A_normalized = normalize_adjacency_matrix(A)

    one_indices = torch.arange(V, device=points.device).unsqueeze(0).repeat(2, 1)
    ones = torch.ones(V, device=points.device)
    L = torch.sparse_coo_tensor(one_indices, ones, (V, V)) - A_normalized

    return L


def point_laplacian_loss(all_points, n_samples=10000, num_neighbors=12):
    """_summary_

    _extended_summary_

    Args:
        sample_points: (N, 3)
        all_points: (M, 3)
        num_neighbors: _description_. Defaults to 12.
    """
    # all_points.retain_grad()
    N = all_points.shape[0]
    n_samples = min(n_samples, N - 1)
    sample_idx = torch.randint(0, N, (n_samples,))
    sample_points = all_points[sample_idx]
    sample_points = sample_points.unsqueeze(0)
    all_points = all_points.unsqueeze(0)  # .detach()

    # Find the nearest neighbors
    dist, idx, nn = knn_points(sample_points, all_points, K=num_neighbors + 1, return_nn=True)

    # nn = nn[0, 1:]  # (N, K, 3)

    new_points = nn.reshape(-1, sample_points.shape[-1])  # (N * (K+1), 3)
    new_indices = torch.arange(1, num_neighbors + 1).unsqueeze(0) + torch.arange(0, n_samples).unsqueeze(1) * (
        num_neighbors + 1
    )  # (N, K)
    new_indices = new_indices.to(sample_points.device)
    # new_points[new_indices.flatten()].requires_grad_(False)

    new_dists = dist[0, :, 1:]  # (N, K)
    with torch.no_grad():
        L = laplacian_matrix(new_points, new_dists, new_indices)
    loss = L.mm(new_points).reshape(n_samples, num_neighbors + 1, 3)
    loss = loss[:, 0]
    loss = loss.norm(dim=1)

    return loss.mean()


if __name__ == "__main__":
    all_points = torch.rand(100000, 3, requires_grad=True, device="cuda")
    # all_points.retain_grad()
    # all_points = all_points + 0.
    loss = point_laplacian_loss(all_points)
    # import torch.nn as nn
    # loss = nn.L1Loss()(all_points.mean() , torch.tensor([100]).cuda())
    loss.backward()
    print(all_points.grad)

    # print(sample_points.grad)
