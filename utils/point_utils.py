import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math


def depths_to_points(view, depthmap):
  c2w = (view.world_view_transform.T).inverse()
  W, H = view.image_width, view.image_height
  fx = W / (2 * math.tan(view.FoVx / 2.0))
  fy = H / (2 * math.tan(view.FoVy / 2.0))
  intrins = torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]]).float().cuda()
  grid_x, grid_y = torch.meshgrid(
    torch.arange(W, device="cuda").float(), torch.arange(H, device="cuda").float(), indexing="xy"
  )
  points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
  rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
  rays_o = c2w[:3, 3]
  points = depthmap.reshape(-1, 1) * rays_d + rays_o
  return points


def depth_to_normal(view, depth):
  """
  view: view camera
  depth: depthmap
  """
  points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
  output = torch.zeros_like(points)
  dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
  dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
  normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
  output[1:-1, 1:-1, :] = normal_map
  return output


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
  inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
  cam_z = ndc_xyz[..., 2:3]
  cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
  cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
  cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
  return cam_xyz


def depth2point_cam(sampled_depth, ref_intrinsic):
  B, N, C, H, W = sampled_depth.shape
  valid_z = sampled_depth
  valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
  valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
  valid_y, valid_x = torch.meshgrid(valid_y, valid_x, indexing="ij")
  # B,N,H,W
  valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
  valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
  ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
  cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H)  # 1, 1, 5, 512, 640, 3
  return ndc_xyz, cam_xyz


def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
  # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
  _, xyz_cam = depth2point_cam(depth_image[None, None, None, ...], intrinsic_matrix[None, ...])
  xyz_cam = xyz_cam.reshape(-1, 3)
  xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[..., 0:1])], axis=-1) @ torch.inverse(
    extrinsic_matrix
  ).transpose(0, 1)
  xyz_world = xyz_world[..., :3]

  return xyz_world


def depth_pcd2normal(xyz):
  hd, wd, _ = xyz.shape
  bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
  top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
  left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
  right_point = xyz[..., 1 : hd - 1, 2:wd, :]
  left_to_right = right_point - left_point
  bottom_to_top = top_point - bottom_point
  xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
  xyz_normal = F.normalize(xyz_normal, p=2, dim=-1)
  xyz_normal = F.pad(xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant", value=0).permute(1, 2, 0)
  return xyz_normal


def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix):
  # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
  # xyz_normal: (H, W, 3)
  xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix)  # (H*W, 3)
  xyz_world = xyz_world.reshape(*depth.shape, 3)
  xyz_normal = depth_pcd2normal(xyz_world)

  return xyz_normal
