import sys

sys.path.append("../code")
import imageio

imageio.plugins.freeimage.download()

import torch
import torch.nn as nn
import numpy as np
import argparse
import imageio
import cv2
import os

from scene.direct_light_sh import DirectLightEnv
from arguments import OptimizationParams
from utils.image_utils import save_image_raw

TINY_NUMBER = 1e-8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    op = OptimizationParams(parser)
    parser.add_argument("--envmap_path", type=str, default="envmaps/envmap12.exr")
    parser.add_argument("--num_shs", type=int, default=128)
    parser.add_argument("--shs_degree", type=int, default=3)
    args = parser.parse_args()

    opt = op.extract(args)
    # load ground-truth envmap
    filename = os.path.abspath(args.envmap_path)
    gt_envmap = imageio.imread(filename)[:, :, :3]
    gt_envmap = cv2.resize(gt_envmap, (512, 256), interpolation=cv2.INTER_AREA)
    gt_envmap = torch.from_numpy(gt_envmap).cuda()
    H, W = gt_envmap.shape[:2]

    out_dir = filename[:-4]
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    assert os.path.isdir(out_dir)

    num_shs = args.num_shs
    shs_degree = args.shs_degree
    direct_light_env = DirectLightEnv(num_shs, shs_degree)
    direct_light_env.training_setup(opt)

    # reload sg parameters if exists

    N_iter = 100000
    for step in range(N_iter):
        env_map = direct_light_env.render_envmap(H, W)
        loss = torch.mean((env_map - gt_envmap) * (env_map - gt_envmap))
        # loss = torch.mean((env_map - gt_envmap).abs())
        loss.backward()
        direct_light_env.step()

        if step % 50 == 0:
            print("step: {}, loss: {}".format(step, loss.item()))

        if step % 100 == 0:
            envmap_check = env_map.clone().detach().cpu().numpy()
            gt_envmap_check = gt_envmap.clone().detach().cpu().numpy()
            diff_map = np.linalg.norm((envmap_check - gt_envmap_check), axis=-1, keepdims=True).repeat(3, axis=-1)
            im = np.concatenate((gt_envmap_check, envmap_check, diff_map), axis=0)
            im = np.clip(np.power(im, 1.0 / 2.2), 0.0, 1.0)
            im = np.uint8(im * 255.0)
            imageio.imwrite(os.path.join(out_dir, "log_im_{}_{}.png".format(num_shs, step)), im)
            
            save_image_raw(os.path.join(out_dir, "envmap_{}_{}.exr".format(num_shs, step)), envmap_check)

            direct_light_env.save_ckpt(out_dir, step)
