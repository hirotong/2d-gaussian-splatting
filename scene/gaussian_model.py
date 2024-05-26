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

import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from arguments import OptimizationParams
from scene.NVDIFFREC import create_trainable_env_rnd, load_env
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    flip_align_view,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(
                torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
                rotation,
            ).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.diffuse_activation = torch.sigmoid
        self.metallic_activation = torch.sigmoid
        self.specular_activation = torch.sigmoid
        self.roughness_activation = torch.sigmoid
        self.albedo_activation = torch.sigmoid
        self.inverse_metallic_activation = inverse_sigmoid
        self.inverse_specular_activation = inverse_sigmoid
        self.inverse_roughness_activation = inverse_sigmoid
        self.inverse_albedo_activation = inverse_sigmoid

    def __init__(self, sh_degree: int, brdf_dim: int, brdf_mode: str, brdf_envmap_res: int):
        if (brdf_dim >= 0 and sh_degree >= 0) or (brdf_dim < 0 and sh_degree < 0):
            raise Exception("Please provide exactly one of either brdf_dim or sh_degree!")

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # brdf setting
        self.brdf = brdf_dim >= 0
        self.brdf_dim = brdf_dim
        self.brdf_mode = brdf_mode
        self.brdf_envmap_res = brdf_envmap_res

        self._normal = torch.empty(0)
        self._normal2 = torch.empty(0)
        self._metallic = torch.empty(0)
        self._specular = torch.empty(0)
        self._roughness = torch.empty(0)
        self._albedo = torch.empty(0)

        self.default_metallic = 0.3
        self.default_specular = [0.5, 0.5, 0.5]
        self.default_roughness = 0.5
        self.roughness_bias = 0.0
        self.default_albedo = [0.3, 0.3, 0.3]

        if self.brdf:
            self.brdf_mlp = create_trainable_env_rnd(self.brdf_envmap_res, scale=0.0, bias=0.8)
        else:
            self.brdf_mlp = None

    def capture(self):
        if self.brdf:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._metallic,
                self._specular,
                self._roughness,
                self._albedo,
                self._normal,
                self._normal2,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )

    # TODO
    def restore(self, model_args, training_args):
        if self.brdf:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._metallic,
                self._specular,
                self._roughness,
                self._albedo,
                self._normal,
                self._normal2,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
        else:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)  # .clamp(max=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def get_normal(self, dir_pp_normalized=None, return_delta=False):
        normal_axis = self.get_normal_axis()
        normal_axis, positive = flip_align_view(normal_axis, dir_pp_normalized)
        delta_normal1 = self._normal  # (N, 3)
        delta_normal2 = self._normal2  # (N, 3)
        delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1)  # (N, 3, 2)
        idx = torch.where(positive, 0, 1).long()[:, None, :].repeat(1, 3, 1)  # (N, 3, 1)
        delta_normal = torch.gather(delta_normal, dim=-1, index=idx).squeeze(-1)  # (N, 3)
        normal = delta_normal + normal_axis
        normal = normal / normal.norm(dim=1, keepdim=True)  # (N, 3)
        if return_delta:
            return normal, delta_normal
        else:
            return normal

    @property
    def get_diffuse(self):
        return self._features_dc

    @property
    def get_metallic(self):
        return self.metallic_activation(self._metallic)

    @property
    def get_specular(self):
        return self.specular_activation(self._specular)

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness + self.roughness_bias)

    @property
    def get_albedo(self):
        return self.albedo_activation(self._albedo)

    @property
    def get_brdf_features(self):
        return self._features_rest

    def get_normal_axis(self):
        # normal axis will always be the z axis
        rotations = self.get_rotation
        R = build_rotation(rotations)
        normal_axis = R[..., 2]
        return normal_axis  # n_points, 3

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if not self.brdf:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0
        elif self.brdf_mode == "envmap":
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            features = torch.zeros((fused_color.shape[0], 3, (self.brdf_dim + 1) ** 2 + 1)).float().cuda()
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0
        else:
            raise NotImplementedError

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.brdf:
            normals = torch.zeros_like(fused_point_cloud, dtype=torch.float, device=self._xyz.device)
            normals2 = torch.zeros_like(fused_point_cloud, dtype=torch.float, device=self._xyz.device)
            self._normal = nn.Parameter(normals.requires_grad_(True))
            self._normal2 = nn.Parameter(normals2.requires_grad_(True))
            self._metallic = nn.Parameter(
                self.inverse_metallic_activation(
                    self.default_metallic * torch.ones((fused_point_cloud.shape[0], 1), device="cuda")
                ).requires_grad_(True)
            )
            self._specular = nn.Parameter(
                self.inverse_specular_activation(torch.tensor(self.default_specular, device="cuda")[None])
                * torch.ones((fused_point_cloud.shape[0], 3), device="cuda").requires_grad_(True)
            )
            self._roughness = nn.Parameter(
                self.inverse_roughness_activation(
                    self.default_roughness * torch.ones((fused_point_cloud.shape[0], 1), device="cuda")
                ).requires_grad_(True)
            )
            self._albedo = nn.Parameter(
                self.inverse_albedo_activation(torch.tensor(self.default_albedo, device="cuda")[None])
                * torch.ones((fused_point_cloud.shape[0], 3), device="cuda").requires_grad_(True)
            )

    def training_setup(self, training_args: OptimizationParams):
        self.fix_brdf_lr = training_args.fix_brdf_lr
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        if self.brdf:
            self._normal.requires_grad_(requires_grad=False)
            l.extend(
                [
                    {
                        "params": list(self.brdf_mlp.parameters()),
                        "lr": training_args.brdf_mlp_lr_init,
                        "name": "brdf_mlp",
                    },
                    {"params": [self._metallic], "lr": training_args.metallic_lr, "name": "metallic"},
                    {"params": [self._roughness], "lr": training_args.roughness_lr, "name": "roughness"},
                    {"params": [self._specular], "lr": training_args.specular_lr, "name": "specular"},
                    {"params": [self._albedo], "lr": training_args.albedo_lr, "name": "albedo"},
                    {"params": [self._normal], "lr": training_args.normal_lr, "name": "normal"},
                ]
            )
            self._normal2.requires_grad_(requires_grad=False)
            l.extend([{"params": [self._normal2], "lr": training_args.normal_lr, "name": "normal2"}])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.brdf_mlp_scheduler_args = get_expon_lr_func(
            lr_init=training_args.brdf_mlp_lr_init,
            lr_final=training_args.brdf_mlp_lr_final,
            lr_delay_mult=training_args.brdf_mlp_lr_delay_mult,
            max_steps=training_args.brdf_mlp_lr_max_steps,
        )

    def training_setup_SHoptim(self, training_args):
        self.fix_brdf_lr = training_args.fix_brdf_lr
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # self.f_rest_scheduler_args = get_const_lr_func(training_args.feature_lr / 20.0)
        if not self.fix_brdf_lr:
            self.f_rest_scheduler_args = get_expon_lr_func(
                lr_init=training_args.feature_lr / 20.0,
                lr_final=training_args.feature_lr_final / 20.0,
                lr_delay_steps=30000,
                lr_delay_mult=training_args.brdf_mlp_lr_delay_mult,
                max_steps=40000,
            )
            # max_steps=training_args.iterations)

    def _update_learning_rate(self, iteration, param):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == param:
                try:
                    lr = getattr(self, f"{param}_scheduler_args", self.brdf_mlp_scheduler_args)(iteration)
                    param_group["lr"] = lr
                    return lr
                except AttributeError:
                    pass

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        self._update_learning_rate(iteration, "xyz")
        if self.brdf and not self.fix_brdf_lr:
            for param in ["brdf_mlp", "roughness", "specular", "normal", "f_dc", "f_rest"]:
                lr = self._update_learning_rate(iteration, param)

    def construct_list_of_attributes(self, brdf_params=True, viewer_fmt=False):
        if brdf_params:
            return self._construct_list_of_attributes_brdf(viewer_fmt)
        else:
            return self._construct_list_of_attributes_gs()

    def _construct_list_of_attributes_gs(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        # for i in range(self._scaling.shape[1]):
        for i in range(3):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def _construct_list_of_attributes_brdf(self, viewer_fmt=False):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        l.extend(["nx0", "ny2", "nz2"])
        # All channels except the 1 DC
        assert self.brdf, "BRDF is not enabled!"
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        if self.brdf_mode == "envmap":
            for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
                l.append("f_rest_{}".format(i))
        else:
            raise NotImplementedError
        # elif self.brdf_mode == "envmap" and self.brdf_dim == -2:
        #   features_rest_len = self._features_rest.shape[-1]
        l.append("opacity")
        for i in range(self._scaling.shape[-1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[-1]):
            l.append("rot_{}".format(i))
        l.append("metallic")
        for i in range(self._specular.shape[-1]):
            l.append("specular_{}".format(i))
        l.append("roughness")
        for i in range(self._albedo.shape[-1]):
            l.append("albedo_{}".format(i))
        return l

    def save_ply(self, path, brdf_params=True, viewer_fmt=False):
        assert not brdf_params or self.brdf, "BRDF is not enabled!"
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz) if not self.brdf else self._normal.detach().cpu().numpy()
        normals2 = self._normal2.detach().cpu().numpy() if (self.brdf) else np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        if brdf_params:
            metallic = self._metallic.detach().cpu().numpy()
            specular = self._specular.detach().cpu().numpy()
            roughness = self._roughness.detach().cpu().numpy()
            albedo = self._albedo.detach().cpu().numpy()

        if viewer_fmt:
            f_dc = 0.5 + (0.5 * normals)
            f_rest = np.zeros((f_rest.shape[0], 45))
            normals = np.zeros_like(normals)

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes(brdf_params, viewer_fmt)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if brdf_params and not viewer_fmt:
            attributes = np.concatenate(
                (
                    xyz,
                    normals,
                    normals2,
                    f_dc,
                    f_rest,
                    opacities,
                    scale,
                    rotation,
                    metallic,
                    specular,
                    roughness,
                    albedo,
                ),
                axis=1,
            )
        else:
            scale = np.concatenate((scale, -10 * np.ones((scale.shape[0], 1))), axis=1)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz)
        pcd_o3d.colors = o3d.utility.Vector3dVector(f_dc)
        return pcd_o3d

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        if not self.brdf:
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        elif self.brdf_mode == "envmap":
            features_extra = np.zeros((xyz.shape[0], 3 * (self.brdf_dim + 1) ** 2))
            if len(extra_f_names) == 3 * (self.brdf_dim + 1) ** 2:
                for idx, attr_name in enumerate(extra_f_names):
                    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            else:
                print("NO INITIAL SH FEATURES FOUND!!! USE ZERO SH AS INITIALIZE.")
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.brdf_dim + 1) ** 2))
        else:
            raise NotImplementedError

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if self.brdf:
            metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]
            specular_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("specular")]
            specular_names = sorted(specular_names, key=lambda x: int(x.split("_")[-1]))
            specular = np.zeros((xyz.shape[0], len(specular_names)))
            for idx, attr_name in enumerate(specular_names):
                specular[:, idx] = np.asarray(plydata.elements[0][attr_name])
            roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
            albedo_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("albedo")]
            albedo_names = sorted(albedo_names, key=lambda x: int(x.split("_")[-1]))
            albedo = np.zeros((xyz.shape[0], len(albedo_names)))
            for idx, attr_name in enumerate(albedo_names):
                albedo[:, idx] = np.asarray(plydata.elements[0][attr_name])

            normal = np.stack(
                (
                    np.asarray(plydata.elements[0]["nx"]),
                    np.asarray(plydata.elements[0]["ny"]),
                    np.asarray(plydata.elements[0]["nz"]),
                ),
                axis=1,
            )
            normal2 = np.stack(
                (
                    np.asarray(plydata.elements[0]["nx2"]),
                    np.asarray(plydata.elements[0]["ny2"]),
                    np.asarray(plydata.elements[0]["nz2"]),
                ),
                axis=1,
            )

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.brdf:
            self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
            self._specular = nn.Parameter(torch.tensor(specular, dtype=torch.float, device="cuda").requires_grad_(True))
            self._roughness = nn.Parameter(
                torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))
            self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
            self._normal2 = nn.Parameter(torch.tensor(normal2, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "brdf_mlp":
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "brdf_mlp":
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.brdf:
            self._metallic = optimizable_tensors["metallic"]
            self._specular = optimizable_tensors["specular"]
            self._roughness = optimizable_tensors["roughness"]
            self._albedo = optimizable_tensors["albedo"]
            self._normal = optimizable_tensors["normal"]
            self._normal2 = optimizable_tensors["normal2"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "brdf_mlp":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_metallic,
        new_specular,
        new_roughness,
        new_albedo,
        new_normal,
        new_normal2,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        if self.brdf:
            d.update(
                {
                    "metallic": new_metallic,
                    "specular": new_specular,
                    "roughness": new_roughness,
                    "albedo": new_albedo,
                    "normal": new_normal,
                    "normal2": new_normal2,
                }
            )

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.brdf:
            self._metallic = optimizable_tensors["metallic"]
            self._specular = optimizable_tensors["specular"]
            self._roughness = optimizable_tensors["roughness"]
            self._albedo = optimizable_tensors["albedo"]
            self._normal = optimizable_tensors["normal"]
            self._normal2 = optimizable_tensors["normal2"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        if torch.sum(selected_pts_mask) == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_metallic = self._metallic[selected_pts_mask].repeat(N, 1) if self.brdf else None
        new_specular = self._specular[selected_pts_mask].repeat(N, 1) if self.brdf else None
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1) if self.brdf else None
        new_albedo = self._albedo[selected_pts_mask].repeat(N, 1) if self.brdf else None
        new_normal = self._normal[selected_pts_mask].repeat(N, 1) if self.brdf else None
        new_normal2 = self._normal2[selected_pts_mask].repeat(N, 1) if (self.brdf) else None
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_metallic,
            new_specular,
            new_roughness,
            new_albedo,
            new_normal,
            new_normal2,
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )
        if torch.sum(selected_pts_mask) == 0:
            return
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_metallic = self._metallic[selected_pts_mask] if self.brdf else None
        new_specular = self._specular[selected_pts_mask] if self.brdf else None
        new_roughness = self._roughness[selected_pts_mask] if self.brdf else None
        new_albedo = self._albedo[selected_pts_mask] if self.brdf else None
        new_normal = self._normal[selected_pts_mask] if self.brdf else None
        new_normal2 = self._normal2[selected_pts_mask] if (self.brdf) else None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_metallic,
            new_specular,
            new_roughness,
            new_albedo,
            new_normal,
            new_normal2,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def set_requires_grad(self, attrib_name, state: bool):
        getattr(self, f"_{attrib_name}").requires_grad = state
