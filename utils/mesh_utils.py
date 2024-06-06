#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import math
import os
from functools import partial

import numpy as np
import open3d as o3d
import torch
import trimesh
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from utils.image_utils import apply_depth_colormap, linear2srgb, srgb2linear, hdr2ldr
from utils.render_utils import save_img_f32, save_img_u8
from scripts.eval_dtu.eval import write_vis_pcd

import sklearn.neighbors as skln


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    cluster_to_keep = min(cluster_to_keep, len(cluster_n_triangles))
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx=viewpoint_cam.image_width / 2,
            cy=viewpoint_cam.image_height / 2,
            fx=viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx / 2.0)),
            fy=viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy / 2.0)),
        )

        extrinsic = np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians: GaussianModel, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.pipe = pipe
        pbr_params = {}
        if pipe.brdf:
            pbr_params["sample_num"] = pipe.sample_num
        self.render = partial(render, pipe=pipe, bg_color=background, dict_params=pbr_params, debug=True)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.viewpoint_stack = []
        if self.gaussians.brdf:
            self.metallics = []
            self.speculars = []
            self.roughness = []
            self.albedos = []
            self.diffuse_maps = []
            self.specular_maps = []
            self.normals_cam = []
        self.images = {"gt": []}

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            gt = viewpoint_cam.original_image[0:3, :, :]
            if render_pkg["hdr"]:
                gt = hdr2ldr(gt)
            self.images["gt"].append(gt.cpu())
            if i == 0:
                for k in render_pkg.keys():
                    if not isinstance(render_pkg[k], torch.Tensor):
                        continue
                    if render_pkg[k].ndim < 3:
                        continue
                    self.images[k] = []
            for k in render_pkg.keys():
                if not isinstance(render_pkg[k], torch.Tensor):
                    continue
                if render_pkg[k].ndim < 3:
                    continue
                elif k in ["delta_normal_norm", "rend_dist"]:
                    self.images[k].append(render_pkg[k].cpu())
                elif "normal" in k:
                    img_k = torch.nn.functional.normalize(render_pkg[k], dim=0)
                    self.images[k].append(img_k.cpu())
                else:
                    self.images[k].append(render_pkg[k].cpu())

        map(lambda x: torch.stack(x, dim=0), self.images.values())
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import focus_point_fn, transform_poses_pca

        torch.cuda.empty_cache()
        c2ws = np.array(
            [np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack]
        )
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        center = focus_point_fn(poses)
        self.radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f"voxel_size: {voxel_size}")
        print(f"sdf_trunc: {sdf_trunc}")
        print(f"depth_truc: {depth_trunc}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size, sdf_trunc=sdf_trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.images["render"][i]
            depth = self.images["surf_depth"][i]

            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1, 2, 0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
        return o3d.mesh
        """

        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2 - mag) * (y / mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, normalmap, viewpoint_cam):
            """
            compute per frame sdf
            """
            new_points = (
                torch.cat([points, torch.ones_like(points[..., :1])], dim=-1) @ viewpoint_cam.full_proj_transform
            )
            z = new_points[..., -1:]
            pix_coords = new_points[..., :2] / new_points[..., -1:]
            mask_proj = ((pix_coords > -1.0) & (pix_coords < 1.0) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(
                depthmap.cuda()[None],
                pix_coords[None, None],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).reshape(-1, 1)
            sampled_rgb = (
                torch.nn.functional.grid_sample(
                    rgbmap.cuda()[None],
                    pix_coords[None, None],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .reshape(3, -1)
                .T
            )
            sampled_normal = (
                torch.nn.functional.grid_sample(
                    normalmap.cuda()[None],
                    pix_coords[None, None],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .reshape(3, -1)
                .T
            )
            sdf = sampled_depth - z
            return sdf, sampled_rgb, sampled_normal, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
            Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                samples = inv_contraction(samples)
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1 / (2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:, 0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:, 0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, normal, mask_proj = compute_sdf_perframe(
                    i,
                    samples,
                    depthmap=self.depthmaps[i],
                    rgbmap=self.rgbmaps[i],
                    normalmap=self.depth_normals[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[:, None]
                # update weight
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = self.radius * 2 / N
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction

        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R + 0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )

        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(
            torch.tensor(np.asarray(mesh.vertices)).float().cuda(),
            inv_contraction=None,
            voxel_size=voxel_size,
            return_rgb=True,
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        for k in self.images:
            if k == "render":
                os.makedirs(os.path.join(path, "renders"), exist_ok=True)
            else:
                os.makedirs(os.path.join(path, k), exist_ok=True)

        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            for k in self.images.keys():
                if k == "render":
                    img_k = self.images[k][idx].permute(1, 2, 0).cpu().numpy()
                    save_img_u8(img_k, os.path.join(path, "renders", "{0:05d}".format(idx) + ".png"))
                elif k == "surf_depth":
                    img_k = apply_depth_colormap(-self.images[k][idx].permute(1, 2, 0))
                    save_img_u8(img_k.cpu().numpy(), os.path.join(path, k, "{0:05d}".format(idx) + ".png"))
                elif k == "delta_normal_norm":
                    img_k = apply_depth_colormap(self.images[k][idx].permute(1, 2, 0))
                    save_img_u8(img_k.cpu().numpy(), os.path.join(path, k, "{0:05d}".format(idx) + ".png"))
                elif "normal" in k:
                    img_k = self.images[k][idx].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
                    save_img_u8(img_k, os.path.join(path, k, "{0:05d}".format(idx) + ".png"))
                elif k == "rend_alpha":
                    img_k = self.images[k][idx].permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy()
                    save_img_u8(img_k, os.path.join(path, k, "{0:05d}".format(idx) + ".png"))
                else:
                    img_k = self.images[k][idx].permute(1, 2, 0).cpu().numpy()
                    save_img_u8(img_k, os.path.join(path, k, "{0:05d}".format(idx) + ".png"))


# Maximum values for bounding box [-1, 1]^3
EMPTY_PCL_DICT = {
    "completeness": np.sqrt(3),
    "accuracy": np.sqrt(3),
    "completeness2": 3,
    "accuracy2": 3,
    "chamfer": 6,
}

EMPTY_PCL_DICT_NORMALS = {
    "normals completeness": -1.0,
    "normals accuracy": -1.0,
    "normals": -1.0,
}


class MeshEvaluator:
    """Mesh evaluation class.

    It handles the mesh evaluation process.

    Args:
        n_points (int): number of points to be used for evaluation
    """

    def __init__(self, n_points=100_000):
        self.n_points = n_points

    def eval_mesh(self, mesh: trimesh.Trimesh, pointcloud_tgt, normals_tgt=None, remove_wall=False, visualize_pcd=True):
        """Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        """
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            if remove_wall:  #! Remove walls and floors
                pointcloud, idx = mesh.sample(2 * self.n_points, return_index=True)
                eps = 0.007
                x_max, x_min = pointcloud_tgt[:, 0].max(), pointcloud_tgt[:, 0].min()
                y_max, y_min = pointcloud_tgt[:, 1].max(), pointcloud_tgt[:, 1].min()
                z_max, z_min = pointcloud_tgt[:, 2].max(), pointcloud_tgt[:, 2].min()

                mask_x = (pointcloud[:, 0] <= x_max) & (pointcloud[:, 0] >= x_min)
                mask_y = pointcloud[:, 1] >= y_min  # floor
                mask_z = (pointcloud[:, 2] <= z_max) & (pointcloud[:, 2] >= z_min)

                mask = mask_x & mask_y & mask_z
                pointcloud_new = pointcloud[mask]
                # Subsample
                idx_new = np.random.randint(pointcloud_new.shape[0], size=self.n_points)
                pointcloud = pointcloud_new[idx_new]
                idx = idx[mask][idx_new]
            else:
                pointcloud, idx = mesh.sample(self.n_points, return_index=True)

            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]

        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        out_dict = self.eval_pointcloud(pointcloud, pointcloud_tgt, normals, normals_tgt, visualize_pcd=visualize_pcd)

        # if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        #     occ = check_mesh_contains(mesh, points_iou)
        #     out_dict["iou"] = compute_iou(occ, occ_tgt)
        # else:
        #     out_dict["iou"] = 0.0

        return out_dict

    def eval_pointcloud(
        self,
        pointcloud,
        pointcloud_tgt,
        normals=None,
        normals_tgt=None,
        thresholds=np.linspace(1.0 / 1000, 1, 1000),
        visualize_pcd=True,
    ):
        """Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        """
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            print("Empty pointcloud / mesh detected!")
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from the predicted point cloud
        completeness, completeness_normals = distance_p2p(pointcloud_tgt, normals_tgt, pointcloud, normals)
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness_mean = completeness.mean()
        completeness2_mean = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(pointcloud, normals, pointcloud_tgt, normals_tgt)
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy_mean = accuracy.mean()
        accuracy2_mean = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2_mean + accuracy2_mean)
        normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
        chamferL1 = 0.5 * (completeness_mean + accuracy_mean)

        # F-Score
        F = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]

        out_dict = {
            "completeness": completeness_mean,
            "accuracy": accuracy_mean,
            "normals completeness": completeness_normals,
            "normals accuracy": accuracy_normals,
            "normals": normals_correctness,
            "completeness2": completeness2_mean,
            "accuracy2": accuracy2_mean,
            "chamfer-L2": chamferL2,
            "chamfer-L1": chamferL1,
            "f-score": F[9],  # threshold = 1.0%
            "f-score-15": F[14],  # threshold = 1.5%
            "f-score-20": F[19],  # threshold = 2.0%
        }

        # visualize error
        if visualize_pcd:
            # vis_dis = 0.05
            # R = np.array([[1, 0, 0]], dtype=np.float64)
            # G = np.array([[0, 1, 0]], dtype=np.float64)
            # B = np.array([[0, 0, 1]], dtype=np.float64)
            # W = np.array([[1, 1, 1]], dtype=np.float64)
            # data_color = np.tile(B, (pointcloud.shape[0], 1))
            # data_alpha = accuracy.clip(max=vis_dis) / vis_dis
            # data_color = R * data_alpha + W * (1 - data_alpha)
            # data_color[np.where(accuracy[:, 0] >= vis_dis)] = G
            # d2s_pcd = o3d.geometry.PointCloud()
            # d2s_pcd.points = o3d.utility.Vector3dVector(pointcloud)
            # d2s_pcd.colors = o3d.utility.Vector3dVector(data_color)

            # gt_color = np.tile(B, (pointcloud_tgt.shape[0], 1))
            # gt_alpha = completeness.clip(max=vis_dis) / vis_dis
            # gt_color = R * gt_alpha + W * (1 - gt_alpha)
            # gt_color[np.where(completeness[:, 0] >= vis_dis)] = G
            # s2d_pcd = o3d.geometry.PointCloud()
            # s2d_pcd.points = o3d.utility.Vector3dVector(pointcloud_tgt)
            # s2d_pcd.colors = o3d.utility.Vector3dVector(gt_color)

            from matplotlib import cm

            colormap = cm.get_cmap("jet")
            # colormap = np.array(colormap.colors)

            error_thresh = 0.1
            error_d2s = accuracy.copy()
            error_d2s = error_d2s.clip(max=error_thresh) / error_thresh  # Normalize to [0, 1]
            error_d2s = (error_d2s * 255).astype(np.uint8)
            error_d2s_vis = colormap(error_d2s[..., 0])
            d2s_pcd = o3d.geometry.PointCloud()
            d2s_pcd.points = o3d.utility.Vector3dVector(pointcloud)
            d2s_pcd.colors = o3d.utility.Vector3dVector(error_d2s_vis[..., :3])

            error_s2d = completeness.copy()
            error_s2d = error_s2d.clip(max=error_thresh) / error_thresh  # Normalize to [0, 1]
            error_s2d = (error_s2d * 255).astype(np.uint8)
            error_s2d_vis = colormap(error_s2d[..., 0])
            s2d_pcd = o3d.geometry.PointCloud()
            s2d_pcd.points = o3d.utility.Vector3dVector(pointcloud_tgt)
            s2d_pcd.colors = o3d.utility.Vector3dVector(error_s2d_vis[..., :3])

            return out_dict, d2s_pcd, s2d_pcd

        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = skln.KDTree(points_tgt)
    dist, idx = kdtree.query(points_src, k=1)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def distance_p2m(points, mesh):
    """Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    """
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def get_threshold_percentage(dist, thresholds):
    """Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]
    return in_threshold
