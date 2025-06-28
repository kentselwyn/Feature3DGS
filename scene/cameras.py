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
import torch
import numpy as np
from torch import nn
from utils.plot import plot_points
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getIntrinsicMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, focal_length,
                 image, gt_alpha_mask,
            image_name, uid, 
            intrinsic_params,
            intrinsic_model,
            trans=np.array([0.0, 0.0, 0.0]), 
            scale=1.0, 
            data_device = "cpu",
            encoder = None,
            mlp = None
        ): 
        super(Camera, self).__init__()
        print(uid)
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.semantic_feature = None
        self.score_feature = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_height = self.original_image.shape[1]
        self.image_width = self.original_image.shape[2]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        _ , H, W = self.original_image.shape
        with torch.no_grad():
            data = {}
            data["image"] = self.original_image.unsqueeze(0).cuda()
            pred = encoder(data)
            kpts = pred["keypoints"][0]
            desc = pred["dense_descriptors"][0]
            x=mlp(desc.permute(1,2,0)).permute(2,0,1)
            score = torch.zeros((1, H, W), dtype=torch.float32).cuda()
            new_img = self.original_image.cuda()
            plot_points(new_img, kpts)
            plot_points(score, kpts)

        # self.original_image = new_img.cpu()
        self.score_feature = score.cpu()
        self.semantic_feature = x.detach().cpu().clone()

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, 
                                                     fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(
                                        self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.extrinsic_matrix = getWorld2View2(R, T, trans, scale)
        
        if intrinsic_params is not None:
            K = getIntrinsicMatrix(intrinsic_params, intrinsic_model)
            self.intrinsic_matrix = K
            self.intrinsic_params = intrinsic_params
        else:
            f = focal_length
            W, H = self.image_width, self.image_height
            cx, cy = W / 2, H / 2
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            self.intrinsic_matrix = K
    
    def update_RT(self, R, t):
        self.R = R
        self.T = t
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, 
                                                     zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.extrinsic_matrix = getWorld2View2(R, t, self.trans, self.scale)

        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.projection_matrix = torch.bmm(self.world_view_transform.unsqueeze(0).inverse(), 
                                           self.full_proj_transform.unsqueeze(0)).squeeze(0)
