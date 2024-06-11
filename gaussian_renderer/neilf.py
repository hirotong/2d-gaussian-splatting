import math
import torch
import numpy as np
import torch.nn.functional as F
from arguments import OptimizationParams, PipelineParams
from bvh import RayTracer
from scene.gaussian_model import GaussianModel
from scene.direct_light_sh import DirectLightEnv
from scene.gs_light import GaussianEnvLighting
from scene.cameras import Camera
from utils.sh_utils import eval_sh, eval_sh_coef
from utils.loss_utils import (
    ssim,
    bilateral_smooth_loss,
    mask_entropy_loss,
    zero_one_loss,
    point_laplacian_loss,
    predicted_normal_loss,
    delta_normal_loss,
)
from utils.image_utils import psnr
from utils.graphics_utils import fibonacci_sphere_sampling, normal_from_depth_image
from diff_surfel_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    RenderEquation,
    RenderEquation_complex,
)

from utils.general_utils import flip_align_view, build_rotation
from utils.point_utils import depth_to_normal


def rendered_world2cam(viewpoint_cam: Camera, normal, alpha, bg_color):
    # normal: (3, H, W), alpha: (H, W), bg_color: (3)
    # normal_cam: (3, H, W)
    _, H, W = normal.shape
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_world = normal.permute(1, 2, 0).reshape(-1, 3)  # (HxW, 3)
    normal_cam = normal_world @ (torch.inverse(extrinsic_matrix[:3, :3]).transpose(0, 1)).T
    # normal_cam = (
    #     torch.cat([normal_world, torch.ones_like(normal_world[..., 0:1])], axis=-1)
    #     @ torch.inverse(torch.inverse(extrinsic_matrix).transpose(0, 1))[..., :3]
    # )
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
        lighting = extract_env_map(pc.env_light, resolution)  # (H, W, 3)
        lighting = lighting.permute(2, 0, 1)  # (3, H, W)
    else:
        raise NotImplementedError

    return lighting


def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None, ...] > 0.0).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)


def render_view(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    is_training=False,
    dict_params=None,
    debug=False,
):
    # direct_light_env_light = dict_params.get("env_light")
    # gamma_transform = dict_params.get("gamma")
    direct_light_env_light = pc.env_light
    gamma_transform = pc.gamma_transform

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # intrinsic = viewpoint_camera.intrinsics

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        # cx=float(intrinsic[0, 2]),
        # cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        # backward_geometry=True,
        # computer_pseudo_normal=True,
        debug=False,
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
    if not is_training:
        if debug:
            normal_axis = pc.get_normal_axis()
            normal_axis, _ = flip_align_view(normal_axis, dir_pp_normalized)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            dir_pp_normalized = F.normalize(
                viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D, dim=-1
            )
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    base_color = pc.get_albedo
    roughness = pc.get_roughness
    metallic = pc.get_metallic
    normal, delta_norm = pc.get_normal(dir_pp_normalized, return_delta=True)
    delta_normal_norm = delta_norm.norm(dim=1, keepdim=True)
    visibility = pc.get_visibility
    incidents = pc.get_indirect  # incident shs
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    if pipe.compute_neilf_python:
        brdf_color, extra_results = rendering_equation_python(
            base_color,
            roughness,
            metallic,
            normal.detach(),
            viewdirs,
            incidents,
            is_training,
            direct_light_env_light,
            visibility,
            sample_num=dict_params["sample_num"],
        )
    else:
        brdf_color, extra_results = rendering_equation(
            base_color,
            roughness,
            metallic,
            normal.detach(),
            viewdirs,
            incidents,
            is_training,
            direct_light_env_light,
            visibility,
            sample_num=dict_params["sample_num"],
        )

    colors_precomp = brdf_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    rendered_pbr, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    val_gamma = None
    if gamma_transform is not None:
        rendered_pbr = gamma_transform.hdr2ldr(rendered_pbr)
        val_gamma = gamma_transform.gamma.item()

    # render extras
    render_extras = {}
    # diffuse_light = extra_results["diffuse_light"]
    # render_extras.update({"diffuse_light": diffuse_light})
    normal_view_normed = 0.5 * normal + 0.5  # [-1, 1] -> [0, 1]
    render_extras.update({"normal_view": normal_view_normed})
    render_extras.update({"delta_normal_norm": delta_normal_norm.repeat(1, 3)})

    if debug:
        normal_axis_normed = 0.5 * normal_axis + 0.5  # [-1, 1] -> [0, 1]
        render_extras.update({"normal_axis": normal_axis_normed})
    if pipe.brdf:
        render_extras.update(
            {
                "diffuse": base_color,
                "metallic": metallic.repeat(1, 3),
                # "specular": specular,
                "roughness": roughness.repeat(1, 3),
                "albedo": base_color,
                # "specular_color": specular_color,
                # "color_delta": color_delta,
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
        "render_pbr": rendered_pbr,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        # "num_rendered": num_rendered,
        # "num_contrib": num_contrib,
    }

    # rendered normal

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    render_normal = allmap[2:5]  # (3, H, W)
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
    out["diffuse_light"] = extra_results["diffuse_light"]
    out["hdr"] = viewpoint_camera.hdr
    out["val_gamma"] = val_gamma

    return out


def calculate_loss(viewpoint_camera: Camera, pc: GaussianModel, render_pkg: dict, opt: OptimizationParams, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    if "val_gamma" in render_pkg:
        tb_dict["val_gamma"] = render_pkg["val_gamma"]
    rendered_image = render_pkg["render"]
    rendered_alpha = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    rendered_normal_view = render_pkg["normal_view"]
    rendered_dist = render_pkg["rend_dist"]
    surf_normal = render_pkg["surf_normal"]
    # pbr
    rendered_pbr = render_pkg["render_pbr"]
    rendered_albedo = render_pkg["albedo"]
    rendered_metallic = render_pkg["metallic"]
    rendered_roughness = render_pkg["roughness"]

    gt_image = viewpoint_camera.original_image.cuda()
    image_mask = (
        viewpoint_camera.gt_alpha_mask.cuda()
        if viewpoint_camera.gt_alpha_mask is not None
        else torch.ones_like(rendered_alpha)
    )

    loss = 0.0
    if opt.lambda_image > 0:
        Ll1 = F.l1_loss(rendered_image, gt_image)
        ssim_val = ssim(rendered_image, gt_image)
        tb_dict["loss_l1"] = Ll1.item()
        tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
        tb_dict["ssim"] = ssim_val.item()
        loss_image = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
        loss = loss + opt.lambda_image * loss_image

    if opt.lambda_pbr > 0:
        Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
        ssim_val_pbr = ssim(rendered_pbr, gt_image)
        tb_dict["loss_l1_pbr"] = Ll1_pbr.item()
        tb_dict["ssim_pbr"] = ssim_val_pbr.item()
        tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
        loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
        loss = loss + opt.lambda_pbr * loss_pbr

    if opt.lambda_predicted_normal > 0:
        loss_predicted_normal = predicted_normal_loss(
            rendered_normal_view, surf_normal.detach(), rendered_alpha.detach()
        )
        tb_dict["loss_predicted_normal"] = loss_predicted_normal.item()
        loss = loss + opt.lambda_predicted_normal * loss_predicted_normal

    if opt.lambda_delta_reg > 0:
        delta_normal_norm = render_pkg["delta_normal_norm"]
        loss_delta_reg = delta_normal_loss(delta_normal_norm, rendered_alpha)
        tb_dict["loss_delta_reg"] = loss_delta_reg.item()
        loss = loss + opt.lambda_delta_reg * loss_delta_reg

    if opt.lambda_mask_zero_one > 0:
        loss_zero_one = zero_one_loss(rendered_alpha)
        tb_dict["loss_mask_zero_one"] = loss_zero_one.item()
        loss = loss + opt.lambda_mask_zero_one * loss_zero_one

    if opt.lambda_mask_entropy > 0 and not torch.allclose(image_mask, torch.ones_like(image_mask)):
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

    if opt.lambda_light > 0:
        diffuse_light = render_pkg["diffuse_light"]
        mean_light = diffuse_light.mean(-1, keepdim=True).expand_as(diffuse_light)
        loss_light = F.l1_loss(diffuse_light, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_albedo > 0:
        value_img = torch.max(gt_image * image_mask, dim=0, keepdim=True)[0]
        shallow_enhance = gt_image * image_mask
        shallow_enhance = 1 - (1 - shallow_enhance) * (1 - shallow_enhance)

        specular_enhance = gt_image * image_mask
        specular_enhance = specular_enhance * specular_enhance

        k = 5
        specular_weight = 1 / (1 + torch.exp(-k * (value_img - 0.5)))
        target_img = specular_weight * specular_enhance + (1 - specular_weight) * shallow_enhance

        loss_albedo = F.l1_loss(target_img, rendered_albedo)
        tb_dict["loss_albedo"] = loss_albedo.item()
        lambda_albedo = (
            opt.lambda_albedo
        )  # * max(0, 1 - float(dict_params["iteration"]) / (opt.base_color_guide_iter_num+1e-8))
        tb_dict["lambda_albedo"] = lambda_albedo

        loss = loss + lambda_albedo * loss_albedo

    if opt.lambda_albedo_smooth > 0:
        loss_albedo_smooth = bilateral_smooth_loss(rendered_albedo, gt_image, image_mask)
        tb_dict["loss_albedo_smooth"] = loss_albedo_smooth.item()
        loss = loss + opt.lambda_albedo_smooth * loss_albedo_smooth

    if opt.lambda_metallic_smooth > 0:
        loss_metallic_smooth = bilateral_smooth_loss(rendered_metallic, gt_image, image_mask)
        tb_dict["loss_metallic_smooth"] = loss_metallic_smooth.item()
        loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth

    if opt.lambda_roughness_smooth > 0:
        loss_roughness_smooth = bilateral_smooth_loss(rendered_roughness, gt_image, image_mask)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth

    if opt.lambda_visibility > 0:
        num = 10000
        means3D = pc.get_xyz
        visibility = pc.get_visibility
        normal = pc.get_normal()
        opacity = pc.get_opacity
        scaling = pc.get_scaling_3d
        rotation = pc.get_rotation
        cov_inv = pc.get_inv_covariance()

        rand_idx = torch.randperm(means3D.shape[0])[:num]
        rand_visibility_shs_view = visibility.transpose(1, 2).view(-1, 1, 4**2)[rand_idx]
        rand_rays_o = means3D[rand_idx]
        rand_rays_d = torch.randn_like(rand_rays_o)
        rand_normal = normal[rand_idx]
        mask = (rand_rays_d * rand_normal).sum(-1) < 0
        rand_rays_d[mask] *= -1
        sample_sh2vis = eval_sh(3, rand_visibility_shs_view, rand_rays_d)
        sample_vis = torch.clamp(sample_sh2vis + 0.5, 0.0, 1.0)
        raytracer = RayTracer(means3D, scaling, rotation)
        trace_results = raytracer.trace_visibility(rand_rays_o, rand_rays_d, means3D, cov_inv, opacity, normal)

        rand_ray_visibility = trace_results["visibility"]
        loss_visibility = F.l1_loss(rand_ray_visibility, sample_vis)
        tb_dict["loss_visibility"] = loss_visibility.item()
        loss = loss + opt.lambda_visibility * loss_visibility

    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    opt: OptimizationParams = False,
    is_training=False,
    dict_params=None,
    debug=False,
):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(
        viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, is_training, dict_params, debug
    )

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, dict_params["iteration"])
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results


def rendering_equation(
    base_color,
    roughness,
    metallic,
    normals,
    viewdirs,
    incidents,
    is_training=False,
    direct_light_env_light=None,
    visibility=None,
    sample_num=24,
):
    if is_training:
        pbr, _, diffuse_light = RenderEquation(
            base_color,
            roughness,
            metallic,
            normals,
            viewdirs,
            incidents,
            direct_light_env_light.get_env_shs,
            visibility,
            sample_num=sample_num,
            is_training=True,
            debug=False,
        )

        extra_results = {
            "pbr": pbr,
            "diffuse_light": diffuse_light,
        }
    else:
        (
            pbr,
            _,
            incident_lights,
            local_incident_lights,
            global_incident_lights,
            incident_visibility,
            diffuse_light,
            local_diffuse_light,
            accum,
            rgb_d,
            rgb_s,
        ) = RenderEquation_complex(
            base_color,
            roughness,
            metallic,
            normals,
            viewdirs,
            incidents,
            direct_light_env_light.get_env_shs,
            visibility,
            sample_num=sample_num,
        )

        extra_results = {
            "incident_lights": incident_lights,
            "local_incident_lights": local_incident_lights,
            "global_incident_lights": global_incident_lights,
            "incident_visibility": incident_visibility,
            "diffuse_light": diffuse_light,
            "local_diffuse_light": local_diffuse_light,
            "accum": accum,
            "rgb_d": rgb_d,
            "rgb_s": rgb_s,
            "pbr": pbr,
        }

    return pbr, extra_results


def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]


def rendering_equation_python(
    base_color,
    roughness,
    metallic,
    normals,
    viewdirs,
    incidents,
    is_training=False,
    direct_light_env_light=None,
    visibility=None,
    sample_num=24,
    view_pos=None,
):
    incident_dirs, incident_areas = sample_incident_rays(normals, is_training, sample_num)

    base_color = base_color.unsqueeze(-2).contiguous()
    roughness = roughness.unsqueeze(-2).contiguous()
    metallic = metallic.unsqueeze(-2).contiguous()
    normals = normals.unsqueeze(-2).contiguous()
    viewdirs = viewdirs.unsqueeze(-2).contiguous()
    
    view_pos = view_pos.unsqueeze(-2).contiguous() if view_pos is not None else None

    deg = int(np.sqrt(visibility.shape[1]) - 1)
    incident_dirs_coef = eval_sh_coef(deg, incident_dirs).unsqueeze(2)
    shs_view = incidents.transpose(1, 2).view(base_color.shape[0], 1, 3, -1)
    shs_visibility = visibility.transpose(1, 2).view(base_color.shape[0], 1, 1, -1)
    local_incident_lights = torch.clamp_min((incident_dirs_coef[..., : shs_view.shape[-1]] * shs_view).sum(-1), 0)
    if isinstance(direct_light_env_light, DirectLightEnv):
        shs_view_direct = direct_light_env_light.get_env_shs.transpose(1, 2).unsqueeze(1)
        global_incident_lights = torch.clamp_min(
            (incident_dirs_coef[..., : shs_view_direct.shape[-1]] * shs_view_direct).sum(-1) + 0.5, 0
        )
    elif isinstance(direct_light_env_light, GaussianEnvLighting):
        global_incident_lights = direct_light_env_light.radiance(view_pos, incident_dirs)
    else:
        global_incident_lights = torch.zeros_like(local_incident_lights, requires_grad=False)

    incident_visibility = torch.clamp(
        (incident_dirs_coef[..., : shs_visibility.shape[-1]] * shs_visibility).sum(-1) + 0.5, 0, 1
    )
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights + global_incident_lights

    def _dot(a, b):
        return (a * b).sum(dim=-1, keepdim=True)  # [H, W, 1, 1]

    def _f_diffuse(base_color, metallic):
        return (1 - metallic) * base_color / np.pi  # [H, W, 1, 3]

    def _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic):
        # used in SG, wrongly normalized
        def _d_sg(r, cos):
            r2 = (r * r).clamp(min=1e-7)
            amp = 1 / (r2 * np.pi)
            sharp = 2 / r2
            return amp * torch.exp(sharp * (cos - 1))

        D = _d_sg(roughness, h_d_n)

        # Fresnel term F
        F_0 = 0.04 * (1 - metallic) + base_color * metallic  # [H, W, 1, 3]
        F = F_0 + (1.0 - F_0) * ((1.0 - h_d_o) ** 5)  # [H, W, S, 3]

        # geometry term V, we use V = G / (4 * cos * cos) here
        def _v_schlick_ggx(r, cos):
            r2 = ((1 + r) ** 2) / 8
            return 0.5 / (cos * (1 - r2) + r2).clamp(min=1e-7)

        V = _v_schlick_ggx(roughness, n_d_i) * _v_schlick_ggx(roughness, n_d_o)  # [H, W, S, 1]

        return D * F * V

    half_dirs = incident_dirs + viewdirs
    half_dirs = F.normalize(half_dirs, dim=-1)

    h_d_n = _dot(half_dirs, normals).clamp(min=0)
    h_d_o = _dot(half_dirs, viewdirs).clamp(min=0)
    n_d_i = _dot(normals, incident_dirs).clamp(min=0)
    n_d_o = _dot(normals, viewdirs).clamp(min=0)
    f_d = _f_diffuse(base_color, metallic)
    f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    rgb_d = (f_d * transport).mean(dim=-2)
    rgb_s = (f_s * transport).mean(dim=-2)
    pbr = rgb_d + rgb_s
    diffuse_light = transport.mean(dim=-2)

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        "incident_visibility": incident_visibility,
        "diffuse_light": diffuse_light,
    }

    return pbr, extra_results
