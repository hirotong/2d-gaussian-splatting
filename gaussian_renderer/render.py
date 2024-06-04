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

import math

import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from arguments import PipelineParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.NVDIFFREC import extract_env_map
from utils.general_utils import flip_align_view
from utils.graphics_utils import normal_from_depth_image
from utils.point_utils import depth_to_normal
from utils.sh_utils import eval_sh


def rendered_world2cam(viewpoint_cam, normal, alpha, bg_color):
    # normal: (3, H, W), alpha: (H, W), bg_color: (3)
    # normal_cam: (3, H, W)
    _, H, W = normal.shape
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_world = normal.permute(1, 2, 0).reshape(-1, 3)  # (HxW, 3)
    normal_cam = (
        torch.cat([normal_world, torch.ones_like(normal_world[..., 0:1])], axis=-1)
        @ torch.inverse(torch.inverse(extrinsic_matrix).transpose(0, 1))[..., :3]
    )
    normal_cam = normal_cam.reshape(H, W, 3).permute(2, 0, 1)  # (H, W, 3)

    background = bg_color[..., None, None]
    normal_cam = normal_cam * alpha[None, ...] + background * (1.0 - alpha[None, ...])

    return normal_cam


def render_normalmap(viewpoint_cam: Camera, depth, bg_color, alpha):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None, None, ...]
    normal_ref = normal_ref * alpha[..., None] + background * (1.0 - alpha[..., None])
    normal_ref = normal_ref.permute(2, 0, 1)

    return normal_ref


# render 360 lighting for a single gaussian
def render_lighting(pc: GaussianModel, resolution=(512, 1024), sampled_index=None):
    if pc.brdf_mode == "envmap":
        lighting = extract_env_map(pc.brdf_mlp, resolution)  # (H, W, 3)
        lighting = lighting.permute(2, 0, 1)  # (3, H, W)
    else:
        raise NotImplementedError

    return lighting


def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None, ...] > 0.0).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    debug=False,
    speed=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = (
            torch.tensor(
                [[W / 2, 0, 0, (W - 1) / 2], [0, H / 2, 0, (H - 1) / 2], [0, 0, far - near, near], [0, 0, 0, 1]]
            )
            .float()
            .cuda()
            .T
        )
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (
            (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]).permute(0, 2, 1).reshape(-1, 9)
        )  # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)  # (N, 3)
    if not speed:
        if debug:
            normal_axis = pc.get_normal_axis()
            normal_axis, _ = flip_align_view(normal_axis, dir_pp_normalized)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    #   pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.brdf:
            color_delta = None
            delta_normal_norm = None
            if pipe.brdf_mode == "envmap":
                gb_pos = pc.get_xyz  # (N, 3)
                view_pos = viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1)  # (N, 3)
                normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True)  # (N, 3)
                delta_normal_norm = delta_normal.norm(dim=1, keepdim=True)

                diffuse = pc.get_diffuse.squeeze(1)  # (N, 3)
                metallic = pc.get_metallic  # (N, 1)
                specular = pc.get_specular  # (N, 3)
                roughness = pc.get_roughness  # (N, 1)
                albedo = pc.get_albedo  # (N, 3)
                color, brdf_pkg = pc.brdf_mlp.shade(
                    gb_pos[None, None, ...],
                    normal[None, None, ...],
                    diffuse[None, None, ...],
                    roughness[None, None, ...],
                    specular[None, None, ...],
                    metallic[None, None, ...],
                    albedo[None, None, ...],
                    view_pos[None, None, ...],
                    mode=pipe.shading,
                )

                colors_precomp = color.squeeze()  # (N, 3)
                diffuse_color = brdf_pkg["diffuse"].squeeze()  # (N, 3)
                specular_color = brdf_pkg["specular"].squeeze()  # (N, 3)

                # TODO: Fix the brdf_feature problem
                if pc.brdf_dim > 0:
                    shs_view = pc.get_brdf_features.transpose(1, 2).view(-1, 3, (pc.brdf_dim + 1) ** 2)
                    dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1)
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(pc.brdf_dim, shs_view, dir_pp_normalized)
                    color_delta = sh2rgb
                    colors_precomp += color_delta

            else:
                raise NotImplementedError

        elif pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if not speed:
        render_extras = {}
        if debug:
            normal_axis_normed = 0.5 * normal_axis + 0.5  # [-1, 1] -> [0, 1]
            render_extras.update({"normal_axis": normal_axis_normed})
        if pipe.brdf:
            normal_view_normed = 0.5 * normal + 0.5  # [-1, 1] -> [0, 1]
            render_extras.update({"normal_view": normal_view_normed})
            if delta_normal_norm is not None:
                render_extras.update({"delta_normal_norm": delta_normal_norm.repeat(1, 3)})
            if debug:
                render_extras.update(
                    {
                        "diffuse": diffuse,
                        "metallic": metallic.repeat(1, 3),
                        "specular": specular,
                        "roughness": roughness.repeat(1, 3),
                        "albedo": albedo,
                        "diffuse_color": diffuse_color,
                        "specular_color": specular_color,
                        "color_delta": color_delta,
                    }
                )

        out_extras = {}
        for k in render_extras.keys():
            if render_extras[k] is None:
                continue
            image = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=render_extras[k],
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )[0]
            out_extras[k] = image

        for k in ["normal", "normal_axis", "normal_view"] if debug else ["normal", "normal_view"]:
            if k in out_extras.keys():
                out_extras[k] = (out_extras[k] - 0.5) * 2.0  # [0, 1] -> [-1, 1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    # rendered normal

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(
        2, 0, 1
    )
    render_normal = render_normal * render_alpha.detach() + bg_color[:, None, None] * (1 - render_alpha.detach())
    normalize_normal_inplace(render_normal, render_alpha[0])

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = render_depth_expected / render_alpha
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    # surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    # surf_normal = surf_normal.permute(2, 0, 1)
    # # remember to multiply with accum_alpha since render_normal is unnormalized.
    # surf_normal = surf_normal * (render_alpha).detach()
    surf_normal = render_normalmap(viewpoint_camera, surf_depth[0], bg_color, render_alpha[0])

    if debug:
        normalize_normal_inplace(out_extras["normal_axis"], render_alpha[0])
        render_normal_cam = rendered_world2cam(viewpoint_camera, render_normal, render_alpha[0], bg_color)
        surf_normal_cam = rendered_world2cam(viewpoint_camera, surf_normal, render_alpha[0], bg_color)
        normal_axis_cam = rendered_world2cam(viewpoint_camera, out_extras["normal_axis"], render_alpha[0], bg_color)
        normal_view_cam = rendered_world2cam(viewpoint_camera, out_extras["normal_view"], render_alpha[0], bg_color)
        out_extras["render_normal_cam"] = render_normal_cam
        out_extras["surf_normal_cam"] = surf_normal_cam
        out_extras["normal_axis_cam"] = normal_axis_cam
        out_extras["normal_view_cam"] = normal_view_cam

    out.update(
        {
            "rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "rend_dist": render_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
        }
    )

    out.update(out_extras)
    return out
