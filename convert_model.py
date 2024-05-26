#!/usr/bin/env python3
# Author: hiro
# Date: 2024-05-26 23:42:26
# LastEditTime: 2024-05-26 23:54:48
# Description:


import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, get_combined_args
from gaussian_renderer import network_gui, render, render_lighting
from scene import GaussianModel, Scene
from scene.NVDIFFREC.light import extract_env_map
from utils.general_utils import safe_state
from utils.image_utils import linear2srgb, psnr, srgb2linear
from utils.system_utils import searchForMaxIteration


if __name__ == "__main__":
    torch.set_num_threads(4)  # Limit number of threads to avoid usage
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = get_combined_args(parser)
    args.save_iterations.append(args.iterations)

    print("Working on " + args.model_path)

    dataset, iteration = model.extract(args), args.iteration
    gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, dataset.brdf_mode, dataset.brdf_envmap_res)
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))

    gaussians.load_ply(
        os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
    )
    gaussians.save_ply(
        os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(iteration), "point_cloud_gs.ply"),
        brdf_params=False,
    )

    # All done
    print("\nConverting complete.")
