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

import numpy as np
import torch
import cv2

def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

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
        img = np.where(
            img <= 0.0031308, 12.92 * img, 1.055 * (img ** (1.0 / 2.4)) - 0.055
        )
    else:
        img = torch.where(
            img <= 0.0031308, 12.92 * img, 1.055 * (img ** (1.0 / 2.4)) - 0.055
        )
    return img
