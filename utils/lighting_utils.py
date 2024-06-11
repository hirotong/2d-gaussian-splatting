#!/usr/bin/env python3
# Author: hiro
# Date: 2024-06-06 00:01:02
# LastEditTime: 2024-06-11 15:38:28
# Description:


from scene.NVDIFFREC import EnvironmentLight
from scene.direct_light_sh import DirectLightEnv
from scene.gs_light import GaussianEnvLighting
from utils.image_utils import save_image_raw
from scene.NVDIFFREC import util


def save_env_map(fn, light):
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    elif isinstance(light, DirectLightEnv):
        color = light.render_envmap(512, 1024)
    elif isinstance(light, GaussianEnvLighting):
        color = light.render_envmap(512, 1024)
    else:
        raise NotImplementedError

    util.save_image_raw(fn, color.detach().cpu().numpy())
