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
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render, render_lighting
from scene import GaussianModel, Scene
from scene.NVDIFFREC.light import extract_env_map
from utils.general_utils import safe_state
from utils.image_utils import linear2srgb, psnr, srgb2linear
from utils.loss_utils import delta_normal_loss, l1_loss, predicted_normal_loss, ssim, zero_one_loss

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
    checkpoint,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, dataset.brdf_mode, dataset.brdf_envmap_res)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

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
            if gaussians.brdf_mode == "envmap":
                gaussians.brdf_mlp.build_mips()

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, debug=False)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        losses_extra = {}
        if pipe.brdf:
            if iteration >= opt.normal_reg_from_iter and iteration < opt.normal_reg_util_iter:
                losses_extra["predicted_normal"] = predicted_normal_loss(
                    render_pkg["normal_view"], render_pkg["surf_normal"], render_pkg["rend_alpha"]
                )
            else:
                losses_extra["predicted_normal"] = torch.tensor(0.0, device="cuda")
            losses_extra["zero_one"] = zero_one_loss(render_pkg["rend_alpha"])
            if iteration >= opt.dist_reg_from_iter and iteration < opt.dist_reg_until_iter:
                losses_extra["dist"] = render_pkg["rend_dist"].mean()
            else:
                losses_extra["dist"] = torch.tensor(0.0, device="cuda")
            if "delta_normal_norm" not in render_pkg.keys() and opt.lambda_delta_reg > 0:
                assert ()
            if "delta_normal_norm" in render_pkg.keys():
                losses_extra["delta_reg"] = delta_normal_loss(render_pkg["delta_normal_norm"], render_pkg["rend_alpha"])

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        for k in losses_extra.keys():
            loss += getattr(opt, f"lambda_{k}") * losses_extra[k]
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

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * losses_extra["dist"].item() + 0.6 * ema_dist_for_log
            # ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_normal_for_log = 0.4 * losses_extra["predicted_normal"] + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )

            # Log and save
            losses_extra["psnr"] = psnr(image, gt_image).mean()
            if tb_writer is not None:
                tb_writer.add_scalar("train_loss_patches/dist_loss", ema_dist_for_log, iteration)
                tb_writer.add_scalar("train_loss_patches/normal_loss", ema_normal_for_log, iteration)

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                losses_extra,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, 1.0, None, True, False),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
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

            # clamp the environment map
            if pipe.brdf and pipe.brdf_mode == "envmap":
                gaussians.brdf_mlp.clamp_(min=0.0, max=10.0)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


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
    Ll1,
    loss,
    losses_extra,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    pipe = renderArgs[0]
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/L1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        for k in losses_extra.keys():
            tb_writer.add_scalar(f"train_loss_patches/{k}_loss", losses_extra[k].item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )
        if tb_writer:
            if pipe.brdf:
                lighting = render_lighting(scene.gaussians, resolution=(512, 1024))
                tb_writer.add_images("lighting", lighting[None], global_step=iteration)

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if pipe.linear:
                        image = linear2srgb(image)
                        gt_image = linear2srgb(gt_image)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap

                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap="turbo")
                        tb_writer.add_images(
                            config["name"] + "_view_{}/depth".format(viewpoint.image_name),
                            depth[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )

                        for k in render_pkg.keys():
                            if render_pkg[k].dim() < 3 or k in ["render", "surf_depth"]:
                                continue
                            elif "normal" in k:
                                image_k = render_pkg[k] * 0.5 + 0.5
                            elif k == "rend_dist":
                                image_k = colormap(render_pkg[k].cpu().numpy()[0])
                            elif pipe.linear and k in ["albedo", "diffse_color", "specular_color"]:
                                image_k = torch.clamp(linear2srgb(render_pkg[k]), min=0.0, max=1.0)
                            else:
                                image_k = torch.clamp(render_pkg[k], min=0.0, max=1.0)

                            tb_writer.add_images(
                                config["name"] + "_view_{}/{}".format(viewpoint.image_name, k),
                                image_k[None],
                                global_step=iteration,
                            )

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
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
        args.start_checkpoint,
    )

    # All done
    print("\nTraining complete.")
