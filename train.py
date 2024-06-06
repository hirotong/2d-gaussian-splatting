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
import sys
import uuid
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from bvh import RayTracer
from gaussian_renderer import network_gui, render_fn_dict  # , render_lighting
from scene import GaussianModel, Scene
from scene.direct_light_sh import DirectLightEnv
from scene.gamma_trans import LearningGammaTransform
from scene.NVDIFFREC.light import extract_env_map, create_trainable_env_rnd
from utils.general_utils import safe_state
from utils.image_utils import linear2srgb, psnr, hdr2ldr
from utils.loss_utils import (
    delta_normal_loss,
    l1_loss,
    point_laplacian_loss,
    predicted_normal_loss,
    ssim,
    visibility_loss,
    zero_one_loss,
)

from utils.sh_utils import eval_sh

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.render_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    """
    Setup PBR components
    """
    pbr_kwargs = dict()
    if pipe.brdf:
        pbr_kwargs["sample_num"] = pipe.sample_num
        if dataset.env_light_type == "shs":
            print("Using global incident light for regularization.")
            assert dataset.global_shs_degree > 0, "Global SH degree must be greater than 0."
            direct_env_light = DirectLightEnv(dataset.num_global_shs, dataset.global_shs_degree)
            direct_env_light.training_setup(opt)

            if args.checkpoint:
                env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
                print("Trying to load global incident light from ", env_checkpoint)
                if os.path.exists(env_checkpoint):
                    direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
                    print("Successfully loaded!")
                else:
                    print("Failed to load!")

            gaussians.env_light = direct_env_light

            # pbr_kwargs["env_light"] = direct_env_light
        elif dataset.env_light_type == "envmap":
            # envmap =
            pass

        if opt.use_ldr_image:
            print("Using learning gamma transform.")
            gamma_transform = LearningGammaTransform(opt.use_ldr_image)
            gamma_transform.training_setup(opt)

            if args.checkpoint:
                gamma_checkpoint = os.path.dirname(args.checkpoint) + "/gamma_" + os.path.basename(args.checkpoint)
                print("Trying to load gamma checkpoint from ", gamma_checkpoint)
                if os.path.exists(gamma_checkpoint):
                    gamma_transform.create_from_ckpt(gamma_checkpoint, restore_optimizer=True)
                    print("Successfully loaded!")
                else:
                    print("Failed to load!")
            # pbr_kwargs["gamma"] = gamma_transform
            gaussians.gamma_transform = gamma_transform

        if opt.finetune_visibility:
            finetune_visibility(gaussians, 1000)
            # gaussians.finetune_visibility()

    """ Prepare render function and bg"""
    render_fn = render_fn_dict[dataset.render_type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_losses_for_log = defaultdict(int)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if pipe.brdf:
            gaussians.set_requires_grad("normal", state=iteration >= opt.normal_reg_from_iter)
            gaussians.set_requires_grad("normal2", state=iteration >= opt.normal_reg_from_iter)
            if dataset.env_light_type == "envmap":
                gaussians.env_light.build_mips()

            gaussians.set_requires_grad("xyz", state=iteration > opt.brdf_only_until_iter)

        # Render
        pbr_kwargs.update({"iteration": iteration})
        render_pkg = render_fn(
            viewpoint_cam, gaussians, pipe, background, opt=opt, is_training=True, dict_params=pbr_kwargs
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # losses_extra = {}
        # if pipe.brdf:
        #     if iteration >= opt.normal_reg_from_iter and iteration < opt.normal_reg_util_iter:
        #         losses_extra["predicted_normal"] = predicted_normal_loss(
        #             render_pkg["normal_view"], render_pkg["surf_normal"], render_pkg["rend_alpha"]
        #         )
        #     else:
        #         losses_extra["predicted_normal"] = torch.tensor(0.0, device="cuda")
        #     losses_extra["zero_one"] = zero_one_loss(render_pkg["rend_alpha"])
        #     if iteration >= opt.dist_reg_from_iter and iteration < opt.dist_reg_until_iter:
        #         losses_extra["dist"] = render_pkg["rend_dist"].mean()
        #     else:
        #         losses_extra["dist"] = torch.tensor(0.0, device="cuda")
        #     if "delta_normal_norm" not in render_pkg.keys() and opt.lambda_delta_reg > 0:
        #         assert ()
        #     if "delta_normal_norm" in render_pkg.keys():
        #         losses_extra["delta_reg"] = delta_normal_loss(render_pkg["delta_normal_norm"], render_pkg["rend_alpha"])

        #     if opt.lambda_visibility > 0:
        #         losses_extra["visibility"] = visibility_loss(gaussians)

        # point laplacian loss
        # if gaussians.get_xyz.requires_grad:
        #     losses_extra["point_laplacian"] = point_laplacian_loss(gaussians.get_xyz)

        # Loss
        # gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # for k in losses_extra.keys():
        #     loss += getattr(opt, f"lambda_{k}") * losses_extra[k]
        # # regularization
        # # lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        # lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        # rend_dist = render_pkg["rend_dist"]
        # rend_normal = render_pkg["rend_normal"]
        # surf_normal = render_pkg["surf_normal"]
        # # normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        # # normal_loss = lambda_normal * (normal_error).mean()
        # dist_loss = lambda_dist * (rend_dist).mean()

        # # loss
        # total_loss = loss + dist_loss  # + normal_loss
        loss = 0.0
        tb_dict = render_pkg["tb_dict"]
        loss += render_pkg["loss"]
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            for k in tb_dict.keys():
                if k in ["psnr", "psnr_pbr", "loss"]:
                    ema_losses_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_losses_for_log[k]

            if iteration % 10 == 0:
                loss_dict = {k: f"{v:.5f}" for k, v in ema_losses_for_log.items()}
                loss_dict["Points"] = f"{len(gaussians.get_xyz)}"
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
                # Log and save
            training_report(tb_writer, iteration, tb_dict, scene, render_fn, pipe, background, dict_params=pbr_kwargs)

            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % args.save_interval == 0 or iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration > opt.brdf_only_until_iter and iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                if pipe.brdf:
                    gaussians.env_light.step()
                    gaussians.gamma_transform.step()

            # clamp the environment map
            if pipe.brdf and dataset.env_light_type == "envmap":
                gaussians.env_light.clamp_(min=0.0, max=10.0)

            if iteration % args.checkpoint_interval == 0 or iteration in checkpoint_iterations:
                gaussians.save_ckpt(scene.model_path, iteration)

                if pipe.brdf:
                    gaussians.env_light.save_ckpt(scene.model_path, iteration)
                    gaussians.gamma_transform.save_ckpt(scene.model_path, iteration)


def finetune_visibility(gaussians: GaussianModel, iterations=1000):
    visibility_sh_lr = 0.01
    optimizer = torch.optim.Adam(
        [
            {"params": [gaussians._visibility_dc], "lr": visibility_sh_lr},
            {"params": [gaussians._visibility_rest], "lr": visibility_sh_lr},
        ]
    )

    means3D = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scaling = gaussians.get_scaling_3d
    rotation = gaussians.get_rotation
    normal = gaussians.get_normal()
    cov3D_inv = gaussians.get_inv_covariance()
    visibility_shs_view = gaussians.get_visibility.transpose(1, 2)
    vis_sh_degree = np.sqrt(visibility_shs_view.shape[-1]) - 1
    rays_o = means3D
    tbar = tqdm(range(iterations), desc="Finetuning visibility shs")
    raytracer = RayTracer(means3D, scaling, rotation)

    for iteration in tbar:
        rays_d = torch.rand_like(rays_o) * 2 - 1
        rays_d = F.normalize(rays_d, p=2, dim=-1)
        mask = (rays_d * normal).sum(dim=-1) < 0
        rays_d[mask] *= -1
        visibility_shs_view = gaussians.get_visibility.transpose(1, 2)
        sample_sh2vis = eval_sh(vis_sh_degree, visibility_shs_view, rays_d)
        sample_vis = torch.clamp(sample_sh2vis + 0.5, 0.0, 1.0)
        trace_results = raytracer.trace_visibility_2dgs(rays_o, rays_d, means3D, cov3D_inv, opacity, normal)
        visibility = trace_results["visibility"]
        loss = F.l1_loss(sample_vis, visibility)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tbar.set_postfix({"loss": loss.item()})


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(
    tb_writer,
    iteration,
    tb_dict,
    scene: Scene,
    renderFunc,
    pipe: PipelineParams,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    opt: OptimizationParams = None,
    is_training=False,
    **kwargs,
):
    if tb_writer:
        for k, v in tb_dict.items():
            tb_writer.add_scalar(f"train_loss_patches/{k}", v, iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0 or iteration in args.test_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )
        # save lighting image
        if tb_writer:
            if pipe.brdf:
                pass
                # TODO: render lighting
                # lighting = render_lighting(scene.gaussians, resolution=(512, 1024))
                # tb_writer.add_images("lighting", lighting[None], global_step=iteration)

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_pbr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = renderFunc(
                        viewpoint,
                        scene.gaussians,
                        pipe,
                        bg_color,
                        scaling_modifier,
                        override_color,
                        opt,
                        is_training,
                        **kwargs,
                    )

                    hdr = render_pkg["hdr"]

                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    image_pbr = render_pkg.get("render_pbr", torch.zeros_like(image))

                    if hdr:
                        image = hdr2ldr(image)
                        gt_image = hdr2ldr(gt_image)
                        image_pbr = hdr2ldr(image_pbr)
                    else:
                        image = torch.clamp(image, min=0.0, max=1.0)
                        gt_image = torch.clamp(gt_image, min=0.0, max=1.0)
                        image_pbr = torch.clamp(image_pbr, min=0.0, max=1.0)

                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap

                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render_pbr".format(viewpoint.image_name),
                            image_pbr[None],
                            global_step=iteration,
                        )

                        for k in render_pkg.keys():
                            if not isinstance(render_pkg[k], torch.Tensor):
                                continue
                            elif k in ["render", "render_pbr"] or render_pkg[k].dim() < 3:
                                continue
                            elif k in ["surf_depth"]:
                                image_k = render_pkg["surf_depth"]
                                norm = image_k.max()
                                image_k = image_k / norm
                                image_k = colormap(image_k.cpu().numpy()[0], cmap="turbo")
                            elif k in ["rend_dist", "delta_normal_norm"]:
                                image_k = colormap(render_pkg[k].cpu().numpy()[0])
                            elif "normal" in k:
                                image_k = render_pkg[k] * 0.5 + 0.5
                            elif k in ["albedo", "diffse_color", "specular_color"]:
                                # TODO: deal with color space
                                image_k = torch.clamp(render_pkg[k], min=0.0, max=1.0)
                            else:
                                image_k = torch.clamp(render_pkg[k], min=0.0, max=1.0)

                            tb_writer.add_images(
                                config["name"] + "_view_{}/{}".format(viewpoint.image_name, k),
                                image_k[None],
                                global_step=iteration,
                            )

                        if iteration == args.test_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                psnr_pbr_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_PBR {}".format(
                        iteration, config["name"], l1_test, psnr_test, psnr_pbr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr_pbr", psnr_pbr_test, iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(4)  # Limit number of threads to avoid usage
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_interval", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=2000)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("-c", "--checkpoint", dest="checkpoint", type=str, default=None)
    # parser.add_argument("-t", "--type", choices=["3dgs", "neilf"], default="3dgs")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
    )

    # All done
    print("\nTraining complete.")
