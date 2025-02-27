import os
import torch
import argparse
import fpsample
import numpy as np
import os.path as osp
import torch.nn.functional as F

from tqdm import tqdm
from plyfile import PlyData

from utils.sh_utils import *
from utils.open3d_utils import *

from registration.metrics import *
from registration.procrustes import *
from encoders.superpoint.mlp import get_mlp_model

"""
For Coarse Registration
python reg_inference.py \
    --scannet_path /home/kentselwyn/Tohoku/experiments/GaussReg/gsreg/ \
    --output_path /home/kentselwyn/Tohoku/Feature3DGS/output/ \
    --num_sample 20000 \
    --output_pcd True

For Fine Registration
python reg_inference.py \
    --scannet_path /home/kentselwyn/Tohoku/experiments/GaussReg/gsreg/ \
    --output_path /home/kentselwyn/Tohoku/Feature3DGS/output/ \
    --coarse_path /home/kentselwyn/Tohoku/experiments/GaussReg/scannet_test_final/estimated_transform.npz \
    --num_sample 20000 \
    --output_pcd True
"""

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet_path', default='ScanNet-GSReg')
    parser.add_argument('--output_path', default='scannet_test_final')
    parser.add_argument('--num_sample', type=int, default=30000)
    parser.add_argument('--coarse_path', type=str, default=None)
    parser.add_argument('--output_pcd', type=bool, default=False)
    return parser

def read_ply_by_opacity(input_path, point_limit, org_transform):
    """
    Extract point cloud from gaussian splatting file.
    """
    plydata = PlyData.read(input_path)

    opacity = np.asarray(plydata.elements[0].data['opacity'])
    opacity = 1 / (1 + np.exp(-opacity))

    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    index_x = (x < np.percentile(x, 95)) * (x > np.percentile(x, 5))
    index_y = (y < np.percentile(y, 95)) * (y > np.percentile(y, 5))
    index_z = (z < np.percentile(z, 95)) * (z > np.percentile(z, 5))
    index = np.where((opacity>0.7) * index_x * index_y * index_z)[0]
    points = np.stack([x,y,z], axis=1)

    if point_limit is not None and index.shape[0] > point_limit:
        fps_samples_idx = fpsample.bucket_fps_kdline_sampling(points[index], point_limit, h=9)
        index = index[fps_samples_idx]

    # DC colors
    features_dc = np.zeros((points.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    point_colors = SH2RGB(features_dc[..., 0])
    point_colors = np.maximum(point_colors, 0)
    point_max_rgb = np.max(point_colors, axis=1)
    point_max_rgb = np.maximum(point_max_rgb, 1)
    point_colors = point_colors / point_max_rgb[:, np.newaxis]
    point_colors = point_colors[index]

    points = points[index]
    if org_transform is not None:
        points = np.matmul(points, org_transform[:3,:3].T) + org_transform[:3,3][None,:]
    center_point = points.mean(0)
    max_length = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    center_point = center_point + np.array([0, 2 * max_length, 0])    

    # Semantic
    count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("semantic_"))
    semantic_features = np.stack([np.asarray(plydata.elements[0][f'semantic_{i}']) for i in range(count)], axis=1) 
    semantic_features = semantic_features[index]
    # Score
    count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("score_"))
    score_features = np.stack([np.asarray(plydata.elements[0][f'score_{i}']) for i in range(count)], axis=1) 
    score_features = score_features[index]

    return points, point_colors, semantic_features, score_features

def load_point_cloud_from_ply(file_name, point_limit, org_transform):
    points, point_colors, semantic_features, score_features = read_ply_by_opacity(file_name, point_limit, org_transform)
    return points, point_colors, semantic_features, score_features

def load_data(ref_file, src_file, num_sample, ref_transform, src_transform, gt_transform):
    ref_points, ref_colors, ref_semantics, ref_scores = load_point_cloud_from_ply(ref_file, num_sample, ref_transform)
    src_points, src_colors, src_semantics, src_scores = load_point_cloud_from_ply(src_file, num_sample, src_transform)
    
    ref_volume = (ref_points[:,0].max() - ref_points[:,0].min()) * (ref_points[:,1].max() - ref_points[:,1].min()) * (ref_points[:,2].max() - ref_points[:,2].min())
    ref_center = (ref_points.max(0) + ref_points.min(0)) / 2
    ref_points = ref_points - ref_center
    src_volume = (src_points[:,0].max() - src_points[:,0].min()) * (src_points[:,1].max() - src_points[:,1].min()) * (src_points[:,2].max() - src_points[:,2].min())
    src_center = (src_points.max(0) + src_points.min(0)) / 2
    src_points = src_points - src_center
    ref_adjust_scale = 1.
    src_adjust_scale = 1.
    if ref_volume > 50:
        ref_adjust_scale = (50 / ref_volume) ** (1/3)
        ref_points = ref_points * ref_adjust_scale
    elif ref_volume < 10:
        ref_adjust_scale = (30 / ref_volume) ** (1/3)
        ref_points = ref_points * ref_adjust_scale
    if src_volume > 50:
        src_adjust_scale = (50 / src_volume) ** (1/3)
        src_points = src_points * src_adjust_scale
    elif src_volume < 10:
        src_adjust_scale = (30 / src_volume) ** (1/3)
        src_points = src_points * src_adjust_scale
    
    data_dict = {
        "ref_points": torch.from_numpy(ref_points.astype(np.float32)),
        "src_points": torch.from_numpy(src_points.astype(np.float32)),
        "ref_colors": torch.from_numpy(ref_colors.astype(np.float32)),
        "src_colors": torch.from_numpy(src_colors.astype(np.float32)),
        "ref_semantics": torch.from_numpy(ref_semantics.astype(np.float32)),
        "src_semantics": torch.from_numpy(src_semantics.astype(np.float32)),
        "ref_scores": torch.from_numpy(ref_scores.astype(np.float32)),
        "src_scores": torch.from_numpy(src_scores.astype(np.float32)),
        "ref_adjust_scale": ref_adjust_scale,
        "src_adjust_scale": src_adjust_scale,
        "ref_center": torch.from_numpy(ref_center.astype(np.float32)),
        "src_center": torch.from_numpy(src_center.astype(np.float32)),
    }
    data_dict["transform"] = torch.from_numpy(gt_transform.astype(np.float32))
    data_dict["ref_transform"] = torch.from_numpy(ref_transform.astype(np.float32))
    data_dict["src_transform"] = torch.from_numpy(src_transform.astype(np.float32))

    return data_dict

def register_set(data_dict, mlp, chunk_size):
    # Get input
    pcd1 = data_dict["ref_points"].cuda()
    pcd2 = data_dict["src_points"].cuda()
    features1 = data_dict["ref_semantics"].cuda()
    features2 = data_dict["src_semantics"].cuda()

    output_dict = {}

    features1 = mlp.decode(features1)
    features2 = mlp.decode(features2)

    device = features1.device

    # Normalize features for faster cosine similarity computation
    features1 = F.normalize(features1, p=2, dim=1).squeeze()
    features2 = F.normalize(features2, p=2, dim=1).squeeze()
    f1_N, _ = features1.shape
    f2_N, _ = features2.shape

    max_indices = torch.zeros(f1_N, dtype=torch.long, device=device)
    max_similarity = torch.full((f1_N,), -float('inf'), device=device)
    for part in range(0, f2_N, chunk_size):
        chunk = features2[part : part + chunk_size]
        # Use matrix multiplication for faster similarity computation
        similarity = torch.mm(features1, chunk.t())
        chunk_max, chunk_indices = similarity.max(dim=1)
        update_mask = chunk_max > max_similarity
        max_similarity[update_mask] = chunk_max[update_mask]
        max_indices[update_mask] = chunk_indices[update_mask] + part
    matched_pcd2 = pcd2[max_indices]

    # Run weighted Procrustes
    estimated_transform = weighted_procrustes(
        src_points=pcd1,
        ref_points=matched_pcd2,
        weights=max_similarity,
        weight_thresh=0.8, # Filter out bad correspondences
        eps=1e-5,
        return_transform=True
    )

    features1 = features1.detach().cpu()
    features2 = features2.detach().cpu()

    output_dict['matched_ref_points'] = pcd1
    output_dict['matched_src_points'] = matched_pcd2
    output_dict['similarity'] = max_similarity
    output_dict['estimated_transform'] = estimated_transform

    return output_dict

def to_cuda(x):
    """
    Move all tensors to cuda.
    """
    if isinstance(x, list):
        x = [to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.cuda()
    return x

def release_cuda(x):
    """
    Release all tensors to item or numpy array.
    """
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x

def register(args):
    output_path = args.output_path

    scannet_path = osp.join(args.scannet_path, 'test')
    scene_list = np.load(osp.join(args.scannet_path, f'test_transformations.npz'), allow_pickle=True)['transformations'].item()

    gt_transform_list = scene_list['gt_transformations_list']
    ref_transform_list = scene_list['ref_transformations_list']
    src_transform_list = scene_list['src_transformations_list']

    # Initialize MLP
    mlp = get_mlp_model(dim=16, type='SP').cuda()

    # Load coarse registration results
    if args.coarse_path is not None:
        coarse_transform_list = np.load(args.coarse_path, allow_pickle=True)['estimated_transform_list'].item()

    rre_list = []
    rte_list = []
    rse_list = []
    estimated_transform_list = {}

    for scene_name in tqdm(gt_transform_list):
        scene_path = osp.join(scannet_path, scene_name)
        ref_ply_path = osp.join(scene_path, 'A/outputs/SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6/point_cloud/iteration_8000/point_cloud.ply')
        src_ply_path = osp.join(scene_path, 'B/outputs/SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6/point_cloud/iteration_8000/point_cloud.ply')
        
        ref_transform = ref_transform_list[scene_name]
        src_transform = src_transform_list[scene_name]

        # Perform fine registration
        if args.coarse_path is not None:
            src_transform = coarse_transform_list[scene_name] @ src_transform

        # Prepare data
        data_dict = load_data(
            ref_file=ref_ply_path,
            src_file=src_ply_path,
            num_sample=args.num_sample,
            ref_transform=ref_transform_list[scene_name],
            src_transform=src_transform,
            gt_transform=gt_transform_list[scene_name]
        )

        ref_points = data_dict['ref_points']
        src_points = data_dict['src_points']
        ref_center = data_dict['ref_center']
        src_center = data_dict['src_center']
        ref_point_colors = data_dict['ref_colors']
        src_point_colors = data_dict['src_colors']
        ref_adjust_scale = data_dict['ref_adjust_scale']
        src_adjust_scale = data_dict['src_adjust_scale']

        gt_transform = data_dict['transform']
        ref_center = data_dict['ref_center']
        src_center = data_dict['src_center']

        if args.output_pcd:
            ref_points_org = ref_points / ref_adjust_scale + ref_center
            src_points_org = src_points / src_adjust_scale + src_center
            ref_point_colors[:, 0] = ref_point_colors[:, 0]/3 + 2/3
            ref_pcd_color = make_open3d_point_cloud(ref_points_org, ref_point_colors)
            ref_pcd_color.estimate_normals()
            src_point_colors[:, 2] = src_point_colors[:, 2]/3 + 2/3
            src_pcd_color = make_open3d_point_cloud(src_points_org, src_point_colors)
            src_pcd_color.estimate_normals()
            if not osp.exists(osp.join(output_path, str(scene_name))):
                os.makedirs(osp.join(output_path, str(scene_name)))
            o3d.io.write_point_cloud(osp.join(output_path, str(scene_name), 'before_ref.ply'), ref_pcd_color)
            o3d.io.write_point_cloud(osp.join(output_path, str(scene_name), 'before_src.ply'), src_pcd_color)

        data_dict = to_cuda(data_dict)
        
        output_dict = register_set(data_dict, mlp, chunk_size=10000)

        data_dict = release_cuda(data_dict)
        output_dict = release_cuda(output_dict)

        ref_center = ref_center.numpy()
        src_center = src_center.numpy()
        
        estimated_transform = output_dict["estimated_transform"].copy()

        if args.output_pcd:
            np.savez(
                osp.join(output_path, str(scene_name), 'correspondence.npz'),
                ref=output_dict['matched_ref_points'] / ref_adjust_scale + ref_center,
                src=output_dict['matched_src_points'] / src_adjust_scale + src_center,
                similarity=output_dict['similarity']
            )
            ref_points_org = ref_points / ref_adjust_scale + ref_center
            src_points_org = src_points / src_adjust_scale + src_center
            ref_point_colors[:, 0] = ref_point_colors[:, 0]/3 + 2/3
            ref_pcd_color = make_open3d_point_cloud(ref_points_org, ref_point_colors)
            ref_pcd_color.estimate_normals()
            src_point_colors[:, 2] = src_point_colors[:, 2]/3 + 2/3
            src_pcd_color = make_open3d_point_cloud(src_points_org, src_point_colors)
            src_pcd_color.estimate_normals()

            estimated_transform_scale = np.zeros_like(estimated_transform)
            estimated_transform_scale[:3,:3] = estimated_transform[:3,:3] / ref_adjust_scale * src_adjust_scale
            estimated_transform_scale[:3, 3] = estimated_transform[:3, 3] / ref_adjust_scale + ref_center - np.matmul(estimated_transform_scale[:3,:3], src_center)
            estimated_transform_scale[3, 3]  = 1.0

            if not osp.exists(osp.join(output_path, str(scene_name))):
                os.makedirs(osp.join(output_path, str(scene_name)))
            o3d.io.write_point_cloud(osp.join(output_path, str(scene_name), 'after_ref.ply'), ref_pcd_color)
            o3d.io.write_point_cloud(osp.join(output_path, str(scene_name), 'after_src.ply'), src_pcd_color)

        estimated_transform_list[str(scene_name)] = estimated_transform_scale      
        # compute error
        rre, rte, rse = compute_registration_error_w_scale(gt_transform, estimated_transform_scale)
        rre_list.append(rre)
        rte_list.append(rte)
        rse_list.append(rse)
        np.savez(os.path.join(output_path, "estimated_transform.npz"), estimated_transform_list = estimated_transform_list)
    np.savez(os.path.join(output_path, "rre_list.npz"), rre_list = np.array(rre_list))
    np.savez(os.path.join(output_path, "rte_list.npz"), rte_list = np.array(rte_list))
    np.savez(os.path.join(output_path, "rse_list.npz"), rse_list = np.array(rse_list))
    print("rre_avg:", np.array(rre_list).mean())
    print("rte_avg:", np.array(rte_list).mean())
    print("rse_avg:", np.array(rse_list).mean())
    print("rre < 5:", (np.array(rre_list) < 5).sum()/np.array(rre_list).shape[0])
    print("rre < 10:", (np.array(rre_list) < 10).sum()/np.array(rre_list).shape[0])
    print("rte < 0.1:", (np.array(rte_list) < 0.1).sum()/np.array(rte_list).shape[0])
    print("rte < 0.2:", (np.array(rte_list) < 0.2).sum()/np.array(rte_list).shape[0])
    print("rse < 0.1:", (np.array(rse_list) < 0.1).sum()/np.array(rse_list).shape[0])
    print("rse < 0.2:", (np.array(rse_list) < 0.2).sum()/np.array(rse_list).shape[0])


def main():
    parser = make_parser()
    args = parser.parse_args()

    register(args=args)

if __name__ == '__main__': main()