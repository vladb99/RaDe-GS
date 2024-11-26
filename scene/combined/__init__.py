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
import json
import random
import torch

from arguments.combined import ModelParams

from scene.combined.gaussian_model_combined import GaussianModelCombined
from scene.data_readers_dynamic import sceneLoadTypeCallbacks

from utils.combined.camera_utils import camera_to_JSON, cameraList_from_camInfosv2
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import searchForMaxIteration


class SceneCombined:
    gaussians: GaussianModelCombined

    def __init__(self, args: ModelParams, gaussians: GaussianModelCombined, load_iteration=None, shuffle=True, resolution_scales=[1.0], duration=None):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.maxtime = duration

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        if args.loader == "nersemble":
            scene_info = sceneLoadTypeCallbacks["Nersemble"](args.source_path, args.source_path, args.eval, duration=duration)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=2)

        # TODO: RaDe-GS doesn't shuffle
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.train_cameras, resolution_scale, args)

            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.test_cameras, resolution_scale, args)

            print("Loading Video Cameras")
            self.video_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.video_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                   "point_cloud",
                                                   "iteration_" + str(self.loaded_iter),))
        elif args.use_random_init:
            # Initialize with random 3D points
            pointcloud = BasicPointCloud(points=torch.randn((len(scene_info.point_cloud.points), 3)) / 20,
                                         colors=torch.randn((len(scene_info.point_cloud.points), 3)), normals=None)
            self.gaussians.create_from_pcd(pointcloud, self.cameras_extent)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getVideoCameras(self, scale=1.0):
        return self.video_cameras[scale]
