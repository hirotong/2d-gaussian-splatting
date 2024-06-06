#!/usr/bin/env python3
# Author: hiro
# Date: 2024-06-06 00:01:02
# LastEditTime: 2024-06-06 00:36:13
# Description:


from scene.NVDIFFREC import EnvironmentLight
from scene.direct_light_sh import DirectLightEnv
from utils.image_utils import save_image_raw
from scene.NVDIFFREC import util


def save_env_map(fn, light):
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    elif isinstance(light, DirectLightEnv):
        color = light.render_envmap(512, 1024)

    util.save_image_raw(fn, color.detach().cpu().numpy())
