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

import json
import os
import random

from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.NVDIFFREC import load_env
from utils.lighting_utils import save_env_map
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.system_utils import mkdir_p, searchForMaxIteration
from scene.direct_light_sh import DirectLightEnv
from scene.gs_light import GaussianEnvLighting
from scene.gamma_trans import LearningGammaTransform

class Scene:
    gaussians: GaussianModel

    def __init__(
        self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval, linear=args.linear
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
            )
            # TODO: load this outside
            if self.gaussians.brdf:
                if args.env_light_type == "envmap":
                    fn = os.path.join(self.model_path, "brdf_mlp", "iteration_" + str(self.loaded_iter), "brdf_mlp.exr")
                    self.gaussians.env_light = load_env(fn, scale=1.0)
                    print(f"Load envmap from: {fn}")
                elif args.env_light_type == "shs":
                    fn = os.path.join(self.model_path, "env_light_ckpt_{}.pth".format(self.loaded_iter))
                    direct_env_light = DirectLightEnv(args.num_global_shs, args.global_shs_degree)
                    direct_env_light.create_from_ckpt(fn)
                    self.gaussians.env_light = direct_env_light
                    print(f"Load shs from: {fn}")
                elif args.env_light_type == "gaussian":
                    fn = os.path.join(self.model_path, "env_light_ckpt_{}.pth".format(self.loaded_iter))
                    gaussian_env_light = GaussianEnvLighting(args.num_global_shs, args.global_shs_degree)
                    gaussian_env_light.create_from_ckpt(fn)
                    self.gaussians.env_light = gaussian_env_light
                    print(f"Load gaussian from: {fn}")
                else:
                    raise NotImplementedError
                
                # gamma transform
                gamma_checkpoint = os.path.join(self.model_path, "gamma_ckpt_{}.pth".format(self.loaded_iter))
                gamma_transform = LearningGammaTransform(use_ldr_image=True)
                gamma_transform.create_from_ckpt(gamma_checkpoint)
                self.gaussians.gamma_transform = gamma_transform
                print(f"Load gamma transform from: {gamma_checkpoint}")

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_gs.ply"), brdf_params=False)
        if self.gaussians.brdf:
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), brdf_params=True)

            env_light_path = os.path.join(self.model_path, "env_light/iteration_{}/envmap.exr".format(iteration))
            mkdir_p(os.path.dirname(env_light_path))
            save_env_map(env_light_path, self.gaussians.env_light)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
