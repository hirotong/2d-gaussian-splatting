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

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import cm


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(depth, cmap="turbo", min=None, max=None):
    near_plane = float(torch.min(depth)) if min is None else min
    far_plane = float(torch.max(depth)) if max is None else max

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)
    return colored_image


def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out


def srgb2linear(img):
    if isinstance(img, np.ndarray):
        img = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    else:
        img = torch.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    return img


def linear2srgb(img):
    if isinstance(img, np.ndarray):
        img = np.where(img <= 0.0031308, 12.92 * img, 1.055 * (img ** (1.0 / 2.4)) - 0.055)
    else:
        img = torch.where(img <= 0.0031308, 12.92 * img, 1.055 * (img ** (1.0 / 2.4)) - 0.055)
    return img


def hdr2ldr(img, scale=0.666667):
    img = img * scale
    # img = 1 - np.exp(-3.0543 * img)  # Filmic
    img = (img * (2.51 * img + 0.03)) / (img * (2.43 * img + 0.59) + 0.14)  # ACES
    return img


def save_image(fn, x: np.ndarray):
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(
                fn,
                np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8),
                compress_level=3,
            )  # Low compression for faster saving
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
        print("WARNING: FAILED to save image %s" % fn)


def save_image_raw(fn, x: np.ndarray):
    try:
        imageio.imwrite(fn, x)
    except:
        print("WARNING: FAILED to save image %s" % fn)
