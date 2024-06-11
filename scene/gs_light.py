import os

import numpy as np
import torch
import torch.nn as nn
import trimesh

import open3d as o3d
from arguments import OptimizationParams
from utils.general_utils import build_rotation
from utils.sh_utils import RGB2SH, eval_sh_coef


class GaussianEnvLighting:
    def __init__(self, subdivisions, max_degree) -> None:
        self.max_degree = max_degree
        self.subdivisions = subdivisions
        # use icosphere to generate the initial gaussians
        icosphere = trimesh.creation.icosphere(subdivisions=subdivisions)
        o3d_icosphere = icosphere.as_open3d
        o3d_icosphere = o3d.t.geometry.TriangleMesh.from_legacy(o3d_icosphere)
        self.icosphere = o3d_icosphere

        scene = o3d.t.geometry.RaycastingScene()
        sphere_id = scene.add_triangles(o3d_icosphere)
        self.scene = scene
        vertices = torch.tensor(icosphere.vertices, dtype=torch.float32).cuda()
        faces = torch.tensor(icosphere.faces, dtype=torch.int32).cuda()
        self._xyz = nn.Parameter(vertices.requires_grad_(True))
        self._faces = faces

        env_shs = torch.zeros(vertices.shape[0], 3, (self.max_degree + 1) ** 2, dtype=torch.float32, device="cuda")
        env_shs_dc = env_shs[:, :, 0:1]
        env_shs_rest = env_shs[:, :, 1:]

        self._env_shs_dc = nn.Parameter(env_shs_dc.transpose(1, 2).contiguous().requires_grad_(True))
        self._env_shs_rest = nn.Parameter(env_shs_rest.transpose(1, 2).contiguous().requires_grad_(True))

    @property
    def get_env_shs(self):
        shs_dc = self._env_shs_dc
        shs_rest = self._env_shs_rest
        return torch.cat((shs_dc, shs_rest), dim=1)

    def training_setup(self, training_args: OptimizationParams):
        if training_args.env_rest_lr < 0:
            training_args.env_rest_lr = training_args.env_lr / 20.0
        l = [
            # {"params": [self._xyz], "lr": training_args.env_lr, "name": "env_xyz"},
            {"params": [self._env_shs_dc], "lr": training_args.env_lr, "name": "env_dc"},
            {"params": [self._env_shs_rest], "lr": training_args.env_rest_lr, "name": "env_rest"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def radiance(self, view_pos, viewdirs):
        pre_shape = viewdirs.shape[:-1]
        viewdirs = viewdirs.reshape(-1, 3)
        # rays_o = view_pos.detach().cpu().clone().numpy().reshape(-1, 3)
        rays_d = viewdirs.detach().cpu().clone().numpy().reshape(-1, 3)
        rays_o = np.zeros_like(rays_d)
        rays = np.concatenate([rays_o, rays_d], axis=-1)
        # rays = rays.squeeze(1)
        rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        inters = self.scene.cast_rays(rays)

        geometry_ids = inters["geometry_ids"].numpy().astype(int)
        triangle_ids = inters["primitive_ids"].numpy().astype(int)
        primitive_uvs = inters["primitive_uvs"].numpy().astype(np.float32)

        primitive_uvs = torch.tensor(primitive_uvs, dtype=torch.float32).cuda()
        primitive_uvs = torch.cat([primitive_uvs, 1 - primitive_uvs.sum(dim=1, keepdim=True)], dim=1)

        # face_idx = self.icosphre.ray.intersects_first(rays_o, rays_d)  # (N,)

        vertex_idx = self._faces[triangle_ids]  # (N, 3)

        shs_coef = eval_sh_coef(self.max_degree, viewdirs)  # (N, (max_degree + 1) ** 2)

        env_shs = self.get_env_shs.transpose(1, 2).contiguous()  # (M, 3, (max_degree + 1) ** 2)

        env_light = torch.clamp_min(
            (env_shs[vertex_idx] * shs_coef[:, None, None] * primitive_uvs[..., None, None]).sum((1, -1)) + 0.5, 0.0
        )

        return env_light.reshape(pre_shape + (3,))

    def render_envmap(self, H, W, upper_hemi=False):
        if upper_hemi:
            theta, phi = torch.meshgrid(
                torch.linspace(0.0 + 0.5 / H, 0.5 - 0.5 / H, H),
                torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W),
                indexing="ij",
            )
        else:
            theta, phi = torch.meshgrid(
                torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H),
                torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W),
                indexing="ij",
            )

        sintheta, costheta = torch.sin(theta * torch.pi), torch.cos(theta * torch.pi)
        sinphi, cosphi = torch.sin(phi * torch.pi), torch.cos(phi * torch.pi)

        viewdirs = torch.stack([sintheta * cosphi, sintheta * sinphi, costheta], dim=-1)  # H, W, 3
        viewdirs = viewdirs.view(-1, 3).cuda()

        env_light = self.radiance(None, viewdirs)

        return env_light.view(H, W, 3)

    def capture(self):
        captured_list = [
            self.subdivisions,
            self.max_degree,
            self._xyz,
            self._faces,
            self._env_shs_dc,
            self._env_shs_rest,
            self.optimizer.state_dict(),
        ]

        return captured_list

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.subdivisions, self.max_degree, self._xyz, self._faces, self._env_shs_dc, self._env_shs_rest, opt_dict) = (
            model_args[:6]
        )

        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter

    def save_ckpt(self, model_path, iteration):
        print("\n[ITER {}] Saving Global Lighting Checkpoint".format(iteration))
        torch.save((self.capture(), iteration), os.path.join(model_path, f"env_light_ckpt_{iteration}.pth"))
