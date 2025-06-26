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
from arguments import ModelParams
import scene_score.dataset_readers as dataset_readers
from utils.system_utils import searchForMaxIteration
from .camera_utils import cameraList_from_camInfos, camera_to_JSON
from encoders.superpoint.superpoint import SuperPoint

class Scene:

    def __init__(self, args : ModelParams, gaussians, load_iteration=None, shuffle=True, 
                 resolution_scales=[1.0], load_feature=True, view_num=None, load_testcam=1,
                 load_test_cams=True, load_ply_with_feature=False, load_ply_with_feature_trained=False): 
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.load_ply_with_feature = load_ply_with_feature

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "train", "poses")):
            scene_info = dataset_readers.readSplitInfo(args.source_path, images=args.images, view_num=view_num)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = dataset_readers.readColmapSceneInfo(path=args.source_path, foundation_model=args.foundation_model, 
                                                          eval=args.eval, images=args.images, view_num=view_num, 
                                                          load_feature = load_feature, load_testcam=load_testcam)
        else:
            assert False, "Could not recognize scene type!"

        # breakpoint()

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
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        conf = {
            "sparse_outputs": True,
            "dense_outputs": True,
            "max_num_keypoints": 512,
            "detection_threshold": 0.01,
        }
        model = SuperPoint(conf).to("cuda").eval()

        for resolution_scale in resolution_scales:
            # if load_feature:
            print("Loading Training Cameras")
            x = load_ply_with_feature or load_ply_with_feature_trained
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args,
                                                                            encoder=model, load_ply_with_feature=x)
            if load_test_cams:
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args,
                                                                               encoder=model, load_ply_with_feature=x)

        if self.loaded_iter:
            if load_ply_with_feature_trained:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "point_cloud_feat.ply"))
            elif load_ply_with_feature:
                self.gaussians.load_ply_feature_grad(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "point_cloud.ply"), 
                                                    semantic_feature_size=16)
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, 
                                           scene_info.semantic_feature_dim, args.speedup) 

    def save(self, iteration):
        if self.load_ply_with_feature:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_feat.ply"))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
