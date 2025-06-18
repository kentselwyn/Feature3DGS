import os
import os.path as osp

import numpy as np

import torch
import torch.nn.functional as F

from utils.pose_utils import solve_pose
from utils.graphics_utils import fov2focal

from scene.kpdetector import KpDetector
from scene.kpdetector import simple_nms
from scene.gaussian_model import GaussianModel

from encoders.superpoint.mlp import get_mlp_model
from encoders.superpoint.superpoint import SuperPoint

def get_intrinsic_matrix(fovx, fovy, W, H):
    focalX = fov2focal(fovx, W)
    focalY = fov2focal(fovy, H)

    K = np.array(
        [
            [focalX,      0, W / 2],
            [     0, focalY, H / 2],
            [     0,      0,     1],
        ],
        dtype=np.float32,
    )
    return K


def sample_gaussians(gaussians: GaussianModel, idx_sampled):
    sampled_gaussians = GaussianModel(3)
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    
    sampled_gaussians._semantic_feature = gaussians._semantic_feature[idx_sampled]
    sampled_gaussians._score_feature = gaussians._score_feature[idx_sampled]
    
    return sampled_gaussians

def topk_match(correlation_matrix, k=1, threshold=0.5):
    N_image = correlation_matrix.shape[-2]
    
    value, index = torch.topk(correlation_matrix, k=k, dim=-1)

    value_flat = value.flatten(1)
    index_flat = index.flatten(1)
    
    mask = value_flat > threshold

    arange_tensor = torch.arange(N_image, device=correlation_matrix.device)

    index_image = arange_tensor[None].repeat(correlation_matrix.shape[0], k)[mask]
    index_gaussian = index_flat[mask]
    value = value_flat[mask]

    return index_image, index_gaussian, value

class STDLoc:
    def __init__(
        self,
        model_path,
        superpoint : SuperPoint,
        gaussians : GaussianModel,

    ):
        # Get Superpoint
        self.superpoint = superpoint.eval().cuda()
        # Get detector
        self.detector = KpDetector(
            self.superpoint.conf.descriptor_dim
        )
        self.detector.load_state_dict(
            torch.load(
                osp.join(model_path, "detector", f"30000_detector.pth")
            )
        )
        self.detector.eval().cuda()
        # Get Landmarks
        sampled_index = torch.load(
            osp.join(model_path, "detector", "sampled_index.pt")
        )
        self.landmarks = sample_gaussians(gaussians, sampled_index)

    @torch.no_grad()
    def loc_sparse(self, query_image, fovx, fovy, ground_truth=None):
        # Get query feature
        fine_resolution = (query_image.shape[2], query_image.shape[3])
        feature_map = self.superpoint({"image" : query_image.cuda()})["dense_descriptors"]
        # feature_map = self.superpoint({"image" : query_image.permute(2, 3, 1).squeeze(0).cuda()})["dense_descriptors"]
        feature_map = F.interpolate(
            feature_map,
            size=fine_resolution,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        feature_map = F.normalize(feature_map, dim=0, p=2)
        
        H, W = feature_map.shape[-2:]

        # Generate Heatmap
        heatmap = self.detector(feature_map)
        # Non-Maximal Suppression
        kp_scores = simple_nms(heatmap, nms_radius=4).flatten()
        _, kp_ids = torch.topk(kp_scores, 1024)
        position_mask = kp_scores > 0
        kp_ids = kp_ids[position_mask[kp_ids]]
        kp_mask = torch.zeros_like(kp_scores, dtype=torch.bool)
        kp_mask[kp_ids] = True

        # Get MLP Decoder
        mlp = get_mlp_model(dim=16, type='SP').cuda()

        # Generate query features
        query_features = feature_map.reshape(feature_map.shape[0], -1)[:, kp_mask]
        # Generate landmark features
        landmark_features = self.landmarks.get_semantic_feature
        landmark_features = mlp.decode(landmark_features).squeeze(1)

        query_features = F.normalize(query_features, dim=0, p=2)
        landmark_features = F.normalize(landmark_features, dim=0, p=2)

        # Sparse Matching
        correlation_matrix = torch.matmul(query_features.T, landmark_features.T)

        image_index, gaussian_index, _ = topk_match(
            correlation_matrix[None], 
            k=1,
            threshold=0
        )

        p2d = torch.stack([torch.arange(H * W) % W, torch.arange(H * W) // W], dim=1)
        p2d = p2d[kp_mask.cpu()][image_index.cpu()].numpy()

        p3d = self.landmarks.get_xyz[gaussian_index].cpu().numpy()

        import pdb; pdb.set_trace()

        # Solve for Pose
        K = get_intrinsic_matrix(fovx, fovy, W, H)
        pose_w2c, inliers = solve_pose(
            p2d + 0.5,
            p3d,
            K,
            "poselib", # Solver
            12.0,      # Reprojection Error
            0.99999,   # Confidence
            1000000,   # Max Iterations
            1000,      # Min Iterations
        )

        return pose_w2c, inliers




        
    