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
import torch.nn.functional as F
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim, predicted_normal_loss, zero_one_loss, mask_entropy_loss, point_laplacian_loss
from utils.point_utils import depth_to_normal
from utils.sh_utils import eval_sh


def render_view(
    viewpoint_camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    **kwargs,
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

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
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

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(
        2, 0, 1
    )

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
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update(
        {
            "rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "rend_dist": render_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
        }
    )

    return rets


def calculate_loss(viewpoint_camera: Camera, pc: GaussianModel, render_pkg: dict, opt: OptimizationParams, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }

    rendered_image = render_pkg["render"]
    rendered_alpha = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    rendered_dist = render_pkg["rend_dist"]
    surf_normal = render_pkg["surf_normal"]

    gt_image = viewpoint_camera.original_image.cuda()
    image_mask = viewpoint_camera.gt_alpha_mask.cuda() if viewpoint_camera.gt_alpha_mask is not None else None

    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    if opt.lambda_predicted_normal > 0:
        loss_predicted_normal = predicted_normal_loss(rendered_normal, surf_normal.detach(), rendered_alpha.detach())
        tb_dict["loss_predicted_normal"] = loss_predicted_normal.item()
        loss = loss + opt.lambda_predicted_normal * loss_predicted_normal

    if opt.lambda_mask_zero_one > 0:
        loss_zero_one = zero_one_loss(rendered_alpha)
        tb_dict["loss_mask_zero_one"] = loss_zero_one.item()
        loss = loss + opt.lambda_mask_zero_one * loss_zero_one

    if opt.lambda_mask_entropy > 0 and image_mask is not None:
        loss_mask_entropy = mask_entropy_loss(rendered_alpha, image_mask)
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_dist > 0 and iteration >= opt.dist_reg_from_iter and iteration < opt.dist_reg_until_iter:
        loss_dist = rendered_dist.mean()
        tb_dict["loss_dist"] = loss_dist.item()
        loss = loss + opt.lambda_dist * loss_dist

    if (
        opt.lambda_point_laplacian > 0
        and iteration >= opt.point_laplacian_reg_from_iter
        and iteration < opt.point_laplacian_reg_until_iter
    ):
        loss_point_laplacian = point_laplacian_loss(pc.get_xyz)
        tb_dict["loss_point_laplacian"] = loss_point_laplacian.item()
        loss = loss + opt.lambda_point_laplacian * loss_point_laplacian

    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    opt: OptimizationParams = None,
    is_training=False,
    dict_params=None,
):
    """
    Render the scene, and calculate the loss if necessary.
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
    results["hdr"] = viewpoint_camera.hdr

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, dict_params["iteration"])
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results
