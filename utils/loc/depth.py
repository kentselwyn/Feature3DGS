import os
import torch
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scene.gaussian.gaussian_model import GaussianModel
from argparse import ArgumentParser
from gaussian_renderer import render
from utils.match.match_img import extract_kpt
from utils.scoremap_vis import one_channel_vis
from utils.graphics_utils import getWorld2View2, fov2focal
from arguments import ModelParams, PipelineParams, get_combined_args


def project_2d_to_3d(keypoints, depth_map, K, w2c) -> torch.Tensor:
    """
    Project 2D keypoints into 3D space.

    Args:
        keypoints (torch.Tensor): 2D keypoints of shape [N, 2] (u, v).
        depth_map (torch.Tensor): Depth map of shape [1, H, W].
        K (torch.Tensor): Intrinsic matrix of shape [3, 3].
        w2c (torch.Tensor): Extrinsic matrix (world-to-camera) of shape [4, 4].

    Returns:
        torch.Tensor: 3D points in world coordinates of shape [N, 3].
    """
    N = keypoints.shape[0]
    H, W = depth_map.shape[1], depth_map.shape[2]
    u = keypoints[:, 0]  # x-coordinates
    v = keypoints[:, 1]  # y-coordinates
    u = u.clamp(0, W - 1)
    v = v.clamp(0, H - 1)
    depth = depth_map[0, v.long(), u.long()]  # Shape: [N]
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=1).T  # Shape: [3, N]
    K_inv = torch.inverse(K)  # Shape: [3, 3]
    points_camera = (K_inv @ (uv1 * depth))  # Shape: [3, N]
    points_camera_h = torch.cat([points_camera, torch.ones((1, N), 
                                                device=points_camera.device)], dim=0)  # Shape: [4, N]
    w2c_inv = torch.inverse(w2c)  # Shape: [4, 4]
    points_world_h = (w2c_inv @ points_camera_h)  # Shape: [4, N]
    points_world = points_world_h[:3].T  # Shape: [N, 3]
    return points_world


def find_depth(model_param:ModelParams, pipe_param:PipelineParams,):
    gaussians = GaussianModel(model_param.sh_degree)
    model_path = model_param.model_path
    gaussians.load_ply(os.path.join(model_path,
                                    "point_cloud",
                                    "iteration_" + '30000',
                                    "point_cloud.ply"))
    # scene = Scene(model_param, gaussians, load_iteration=-1, shuffle=False)
    bg_color = [1,1,1] if model_param.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # views = scene.getTestCameras()
    # view = views[0]
    view = torch.load(f'{model_path}/camera_seq-03-frame-000000.pt').cuda()
    R = view.R
    T = view.T
    w2c = torch.tensor(getWorld2View2(R, T))
    K = np.eye(3)
    focal_length = fov2focal(view.FoVx, view.image_width)
    K[0, 0] = K[1, 1] = focal_length
    K[0, 2] = view.image_width / 2
    K[1, 2] = view.image_height / 2
    K = torch.tensor(K, dtype=torch.float32)
    render_pkg = render(view, gaussians, pipe_param, background)
    ply_path = "/home/koki/code/cc/feature_3dgs_2/data/cluster_centers.ply"
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']
    positions = np.array([(vertex['x'], vertex['y'], vertex['z']) for vertex in vertex_data])
    colors = np.array([(vertex['red'], vertex['green'], vertex['blue']) for vertex in vertex_data])
    depth = render_pkg["depth"].cpu()
    score_map = render_pkg["score_map"].cpu()
    depth_img = one_channel_vis(depth)
    depth_img.save('depth_median.png')
    centroids = extract_kpt(score_map, 0.5)
    centroids = centroids[:, [1, 0]]
    points_world = project_2d_to_3d(centroids, depth, K, w2c).cpu().detach().numpy()
    center_pcd = o3d.geometry.PointCloud()
    center_pcd.points = o3d.utility.Vector3dVector(points_world)
    center_pcd.paint_uniform_color([0, 0, 1])
    o3d.io.write_point_cloud("points_world_depth_median_adjust.ply", center_pcd)



if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    args = get_combined_args(parser)
    find_depth(Model_param.extract(args), Pipe_param.extract(args))
