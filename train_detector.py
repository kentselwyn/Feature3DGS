import os
import sys
import uuid
import math

import numpy as np

from tqdm import tqdm
from random import randint
from argparse import ArgumentParser, Namespace

from arguments import get_combined_args
from arguments import ModelParams, PipelineParams, OptimizationParams

from scene import Scene
from scene.kpdetector import KpDetector
from scene.gaussian_model import GaussianModel

from utils.loss_utils import bce_loss
from utils.image_utils import get_resolution_from_longest_edge
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov, fov2focal

from encoders.superpoint.mlp import get_mlp_model, get_mlp_dataset
from encoders.superpoint.superpoint import SuperPoint

from gaussian_renderer import get_render_visible_mask, render

import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import sklearn

import torch.nn as nn

def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature

def calculate_matching_score(
    cam,
    mlp,
    gt_fmap,
    gaussians: GaussianModel,
    visible_gaussians,
):
    # Create intrinsic matrix
    focalX = fov2focal(cam.FoVx, gt_fmap.shape[2])
    focalY = fov2focal(cam.FoVy, gt_fmap.shape[1])
    # print("focal:", focalX, focalY)
    intrinsic_matrix = torch.tensor(
        [
            [focalX, 0.0, gt_fmap.shape[2] / 2],
            [0.0, focalY, gt_fmap.shape[1] / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    
    xyz = gaussians.get_xyz
    gaussian_features = gaussians.get_semantic_feature

    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=-1)
    xyz_cam = (torch.from_numpy(cam.extrinsic_matrix).cuda() @ xyz_homo.T)[:3]
    xyz_cam = xyz_cam / xyz_cam[2]

    xy = (intrinsic_matrix @ xyz_cam)[:2].long()

    in_mask = (
        (xy[0] >= 0) & (xy[0] < gt_fmap.shape[2]) &
        (xy[1] >= 0) & (xy[1] < gt_fmap.shape[1])
    )

    if visible_gaussians is not None:
        visible_mask = in_mask & visible_gaussians
    else:
        visible_mask = in_mask
    
    xy = xy[:, visible_mask]
    gaussian_features = gaussian_features[visible_mask, :]
    gaussian_features = F.normalize(gaussian_features, dim=0, p=2)

    image_features = gt_fmap[:, xy[1], xy[0]].T

    # Decode features
    gaussian_features = mlp.decode(gaussian_features).squeeze(1)
    score = (gaussian_features * image_features).sum(-1)

    return score, visible_mask

def cdist2(x, y):
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
    x_sq_norm = x.pow(2).sum(dim=-1)
    y_sq_norm = y.pow(2).sum(dim=-1)
    x_dot_y = x @ y.t()
    sq_dist = x_sq_norm.unsqueeze(dim=1) + y_sq_norm.unsqueeze(dim=0) - 2*x_dot_y
    # For numerical issues
    sq_dist.clamp_(min=0.0)
    return torch.sqrt(sq_dist)

def random_knn_score_2(points, num_points, score, k=32, batch_size=1024):
    sampled_index = torch.randperm(points.shape[0])[:num_points]
    sampled_points = points[sampled_index]
    
    # Split into batches (prevents CUDA Out-of-Memory)    
    num_batches = math.ceil(points.shape[0] / batch_size)
    # Rolling buffer
    accumulated_indices = torch.zeros((num_points, k * 2)).to(device=points.device)
    accumulated_distances = torch.full((num_points, k * 2), float("inf")).to(device=points.device)
    with torch.no_grad():
        for batch_index in tqdm(range(num_batches), position=1, desc="KNN Sampling"):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, points.shape[0])
            sampled_batch = points[start_index:end_index]

            # Calculate distance
            distance = sampled_points.unsqueeze(1) - sampled_batch.unsqueeze(0)
            distance = torch.sum(distance * distance, dim=-1)

            smallest_distances, smallest_indices = torch.topk(distance, k=k, dim=-1, largest=False, sorted=True)
            smallest_indices = start_index + smallest_indices

            if batch_index == 0:
                accumulated_indices[:, :k] = smallest_indices
                accumulated_distances[:, :k] = smallest_distances
                continue
            
            accumulated_indices[:, k:] = smallest_indices
            accumulated_distances[:, k:] = smallest_distances

            index = torch.argsort(
                accumulated_distances,
                descending=False,
                dim=-1
            )
            accumulated_indices = accumulated_indices[torch.arange(accumulated_indices.shape[0]).unsqueeze(1), index]
            accumulated_distances = accumulated_distances[torch.arange(accumulated_indices.shape[0]).unsqueeze(1), index]
        
        accumulated_indices = accumulated_indices[:, :k].to(int)
        index = torch.argsort(
            score[accumulated_indices.to(int).flatten()].reshape(accumulated_indices.shape), 
            descending=True, 
            dim=-1
        )
        accumulated_indices = accumulated_indices[torch.arange(accumulated_indices.shape[0]).unsqueeze(1), index]

        final_sampled_index = set()
        for i in range(num_points):
            for j in accumulated_indices[i]:
                j = j.item()
                if j in final_sampled_index:
                    continue
                final_sampled_index.add(j)
                break
        return torch.tensor(list(final_sampled_index), dtype=torch.int).cuda()

    def random_knn_score(points, num_points, score, k=32, batch_size=2048):
        sampled_index = torch.randperm(points.shape[0])[:num_points]
        sampled_points = points[sampled_index]
        
        # Split into batches (prevents CUDA Out-of-Memory)    
        num_batches = math.ceil(points.shape[0] / batch_size)
        # Rolling buffer
        accumulated_indices = torch.zeros((num_points, k * 2)).to(device=points.device)
        for batch_index in tqdm(range(num_batches), position=1, desc="KNN Sampling"):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, points.shape[0])
            sampled_batch = points[start_index:end_index]

            # Calculate distance
            distance = sampled_points.unsqueeze(1) - sampled_batch.unsqueeze(0)
            distance = torch.sum(distance * distance, dim=-1)

            _, smallest_indices = torch.topk(distance, k=k, dim=-1, largest=False, sorted=True)
            smallest_indices = start_index + smallest_indices

            if batch_index == 0:
                accumulated_indices[:, :k] = smallest_indices
            else:
                accumulated_indices[:, k:] = smallest_indices

            if batch_index != 0:
                index = torch.argsort(
                    score[accumulated_indices.to(int).flatten()].reshape(accumulated_indices.shape), 
                    descending=True, 
                    dim=-1
                )
                accumulated_indices = accumulated_indices[torch.arange(accumulated_indices.shape[0]).unsqueeze(1), index]

    accumulated_indices = accumulated_indices[:, :k].to(int)
    final_sampled_index = set()
    for i in range(num_points):
        for j in accumulated_indices[i]:
            j = j.item()
            if j in final_sampled_index:
                continue
            final_sampled_index.add(j)
            break
    import pdb; pdb.set_trace()
    return torch.tensor(list(final_sampled_index), dtype=torch.int).cuda()

def generate_heatmap(
    cam,
    gaussians, 
    gt_fmap, 
    sampled_index,
    visible_gaussians,
):
    if visible_gaussians is not None:
        visible_gaussian = visible_gaussians[sampled_index]
        sampled_index = sampled_index[visible_gaussian]
    
    sampled_xyz = gaussians.get_xyz[sampled_index]

    # Initialize 
    gt_heatmap = torch.zeros(
        (1, gt_fmap.shape[1], gt_fmap.shape[2]),
        device=gt_fmap.device,
    )

    xyz_homo = torch.cat(
        [sampled_xyz, torch.ones(sampled_xyz.shape[0], 1, device=sampled_xyz.device)],
        dim=-1
    )    
    xyz_cam = (torch.from_numpy(cam.extrinsic_matrix).cuda() @ xyz_homo.T)[:3]
    xyz_cam = xyz_cam / xyz_cam[2]

    # Create intrinsic matrix
    focalX = fov2focal(cam.FoVx, gt_fmap.shape[2])
    focalY = fov2focal(cam.FoVy, gt_fmap.shape[1])
    # print("focal:", focalX, focalY)
    intrinsic_matrix = torch.tensor(
        [
            [focalX, 0.0, gt_fmap.shape[2] / 2],
            [0.0, focalY, gt_fmap.shape[1] / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    xy = (intrinsic_matrix @ xyz_cam)[:2].long()

    in_mask = (
        (xy[0] >= 0) & (xy[0] < gt_fmap.shape[2]) &
        (xy[1] >= 0) & (xy[1] < gt_fmap.shape[1])
    )

    xy_pos = xy[:, in_mask]
    gt_heatmap[:, xy_pos[1], xy_pos[0]] = 1
    
    return gt_heatmap

# Experiment 1
def matching_oriented_sample(
    scene: Scene,
    gaussians : GaussianModel,
    superpoint: SuperPoint,
    pipe: PipelineParams,
    mlp,
    gaussian_mask,
    render_visible_masks,
    landmark_num=16384,
    landmark_k=32,
):
    num_gaussians = gaussians.get_xyz.shape[0]

    viewpoint_stack = scene.getTrainCameras().copy()
    
    score_num =torch.zeros(num_gaussians, dtype=torch.int, requires_grad=False, device="cuda")
    score_sum = torch.zeros(num_gaussians, dtype=torch.float32, requires_grad=False, device="cuda")
    fine_resolution = (viewpoint_stack[0].original_image.shape[1], viewpoint_stack[0].original_image.shape[2])

    for viewpoint_cam in tqdm(viewpoint_stack, position=0, desc="Match Score Calculation"):
        # Get ground truth information
        # Experiment 
        torch.cuda.empty_cache()

        # print(f"\r{it:05d} / {len(viewpoint_stack):05d}", end="")

        with torch.no_grad():
            gt_img = viewpoint_cam.original_image.cuda()
            gt_fmap = superpoint({"image" : gt_img.unsqueeze(0)})["dense_descriptors"]
            gt_fmap = F.interpolate(
                gt_fmap,
                size=fine_resolution,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            gt_fmap = F.normalize(gt_fmap, dim=0, p=2)

        if render_visible_masks.get(viewpoint_cam.image_name, None) is None:
            render_visible_mask = get_render_visible_mask(
                viewpoint_camera=viewpoint_cam,
                pc=gaussians,
                pipe=pipe,
                bg_color=torch.zeros(3, device="cuda"),
            )
            render_visible_masks[viewpoint_cam.image_name] = render_visible_mask
        
        with torch.no_grad():
            score, mask = calculate_matching_score(
                cam=viewpoint_cam,
                mlp=mlp,
                gt_fmap=gt_fmap,
                gaussians=gaussians,
                visible_gaussians=(
                    render_visible_masks[viewpoint_cam.image_name] & gaussian_mask
                        if gaussian_mask is not None else render_visible_masks[viewpoint_cam.image_name]
                ),
            )
            score_num[mask] += 1
            score_sum[mask] += score

    score_num[score_num == 0] = 1
    score_average = score_sum / score_num

    sampled_indexes = random_knn_score_2(gaussians.get_xyz, landmark_num, score_average, landmark_k)
    sampled_indexes = torch.unique(sampled_indexes)
    return sampled_indexes, gaussian_mask, score_average, score_num

# Experiment 2
def matching_oriented_sample_w_score_filter(
    scene: Scene,
    gaussians : GaussianModel,
    superpoint: SuperPoint,
    pipe: PipelineParams,
    mlp,
    render_visible_masks,
    landmark_num=16384,
    landmark_k=32,
    score_threshold=0.8,
):
    gaussian_mask = (gaussians.get_score_feature * gaussians.get_opacity.unsqueeze(-1) >= score_threshold).flatten()

    return matching_oriented_sample(
        scene=scene,
        gaussians=gaussians,
        superpoint=superpoint,
        pipe=pipe,
        mlp=mlp,
        gaussian_mask=gaussian_mask,
        render_visible_masks=render_visible_masks,
        landmark_num=landmark_num,
        landmark_k=landmark_k,
    )

def evaluate_detector(
    model_param : ModelParams,
    pipe_param : PipelineParams,
    gaussians : GaussianModel,
    scene : Scene,
    detector,
    superpoint,
    sampled_idx,
    render_visible_masks,
    tb_writer,
    iteration,
):
    torch.cuda.empty_cache()

    # Sample the Gaussians
    sampled_gaussians = GaussianModel(
        gaussians.max_sh_degree,
    )
    sampled_gaussians._xyz = gaussians.get_xyz[sampled_idx]
    sampled_gaussians._features_dc = gaussians._features_dc[sampled_idx]
    sampled_gaussians._features_rest = gaussians._features_rest[sampled_idx]
    sampled_gaussians._scaling = gaussians.get_scaling[sampled_idx]
    sampled_gaussians._rotation = gaussians.get_rotation[sampled_idx]
    sampled_gaussians._opacity = gaussians.get_opacity[sampled_idx]

    sampled_gaussians._semantic_feature = gaussians.get_semantic_feature[sampled_idx]
    sampled_gaussians._score_feature = gaussians.get_score_feature[sampled_idx]

    validation_configs = (
        {
            "name": "test",
            "cameras": scene.getTestCameras(),
        },
        {
            "name": "train",
            "cameras": [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)
            ],
        }
    )

    for config in validation_configs:
        if (config["cameras"] is not None) and (len(config["cameras"]) > 0):
            fine_resolution = get_resolution_from_longest_edge(
                config["cameras"][0].original_image.shape[1],
                config["cameras"][0].original_image.shape[2],
                scene.longest_edge
            )

            loss_sum = 0.0
            for idx, viewpoint_cam in enumerate(config["cameras"]):
                gt_img = viewpoint_cam.original_image.cuda()
                gt_fmap = superpoint({"image" : gt_img.unsqueeze(0)})["dense_descriptors"]
                gt_fmap = F.interpolate(
                    gt_fmap,
                    size=fine_resolution,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0)
                gt_fmap = F.normalize(gt_fmap, dim=0, p=2)

                if render_visible_masks.get(viewpoint_cam.image_name, None) is None:
                    render_visible_mask = get_render_visible_mask(
                        viewpoint_camera=viewpoint_cam,
                        pc=gaussians,
                        pipe=pipe_param,
                        bg_color=torch.zeros(3, device="cuda"),
                    )
                    render_visible_masks[viewpoint_cam.image_name] = render_visible_mask

                with torch.no_grad():
                    gt_heatmap = generate_heatmap(
                        viewpoint_cam,
                        gaussians, 
                        gt_fmap, 
                        sampled_idx,
                        render_visible_masks[viewpoint_cam.image_name],
                    )

                heatmap = detector(gt_fmap)
                loss = bce_loss(heatmap, gt_heatmap)
                loss_sum += loss.item()

                from PIL import Image
                from utils.vis_scoremap import one_channel_vis, overlay_scoremap

                render_path = os.path.join(scene.model_path, pipe_param.checkpoint_dir, "detector")
                heatmap_vis = overlay_scoremap(
                    (gt_img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8),
                    (gt_heatmap.permute(1,2,0).squeeze().cpu().numpy() * 255).astype(np.uint8),
                )
                heatmap_vis.save(os.path.join(render_path, f'{config["name"]}-{iteration:05d}-{idx:05d}' + "_gt_heatmap_vis.png"))
                heatmap_vis = one_channel_vis(heatmap)
                heatmap_vis.save(os.path.join(render_path, f'{config["name"]}-{iteration:05d}-{idx:05d}' + "_heatmap_vis.png"))
            
            loss_sum /= len(config["cameras"])
            print(f"[ITER {iteration + 1}] Validation loss on {config['name']} set: {loss_sum:.3f}")

            if tb_writer:
                tb_writer.add_scalar(
                    f"detector_loss_patches/{config['name']}_loss", loss_sum, iteration + 1
                )

def training_detector(
    model_param : ModelParams,
    opt_param : OptimizationParams, 
    pipe_param : PipelineParams,
    gaussians : GaussianModel,
    scene: Scene,
    mlp,
    testing_iterations,
    saving_iterations,
    tb_writer,
    score_threshold=0.8,
    train_iteration=30_000,
    landmark_num=16384,
    landmark_k=32,
):
    detector_dir = os.path.join(
        pipe_param.checkpoint_dir, "detector"
    )

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": 1024,
        "detection_threshold": 0.01,
    }
    model = SuperPoint(conf).to("cuda").eval()

    # For caching the rendered visible mask
    render_visible_masks = {}

    if score_threshold is not None:
        sampled_index, _, _, _ = matching_oriented_sample_w_score_filter(
            scene=scene,
            gaussians=gaussians,
            superpoint=model,
            pipe=pipe_param,
            mlp=mlp,
            render_visible_masks=render_visible_masks,
            landmark_num=landmark_num,
            landmark_k=landmark_k,
            score_threshold=score_threshold,
        )
    else:
        sampled_index, _, _, _ = matching_oriented_sample(
            scene=scene,
            gaussians=gaussians,
            superpoint=model,
            pipe=pipe_param,
            mlp=mlp,
            gaussian_mask=None,
            render_visible_masks=render_visible_masks,
            landmark_num=landmark_num,
            landmark_k=landmark_k,
        )

    save_path = os.path.join(scene.model_path, detector_dir)
    os.makedirs(save_path, exist_ok=True)
    torch.save(sampled_index, os.path.join(save_path, "sampled_index.pt"))

    print("Start training detector")
    
    viewpoint_stack = None

    grad_accumulation = 8
    detector = KpDetector(model.conf.descriptor_dim).cuda().train()
    optimizer = torch.optim.AdamW(detector.parameters())
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_iteration // grad_accumulation, 
        eta_min=0.0005
    )

    # progress_bar = tqdm(range(0, train_iteration), desc="Scene-Specific Detector")
    print("Training detector...")
    for iteration in tqdm(range(train_iteration), position=0, desc="Scene-Specific Detector"):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        fine_resolution = get_resolution_from_longest_edge(viewpoint_cam.original_image.shape[1], viewpoint_cam.original_image.shape[2], scene.longest_edge)

        # Get ground truth information
        gt_img = viewpoint_cam.original_image.cuda()
        gt_fmap = model({"image" : gt_img.unsqueeze(0)})["dense_descriptors"]
        gt_fmap = F.interpolate(
            gt_fmap,
            size=fine_resolution,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        gt_fmap = F.normalize(gt_fmap, dim=0, p=2)

        if render_visible_masks.get(viewpoint_cam.image_name, None) is None:
            render_visible_mask = get_render_visible_mask(
                viewpoint_camera=viewpoint_cam,
                pc=gaussians,
                pipe=pipe_param,
                bg_color=torch.zeros(3, device="cuda"),
            )
            render_visible_masks[viewpoint_cam.image_name] = render_visible_mask

        # Predict heatmap from feature map using detector
        pred_heatmap = detector(gt_fmap)
        # Calculate the ground truth heatmap from the Gaussians
        with torch.no_grad():
            gt_heatmap = generate_heatmap(
                viewpoint_cam,
                gaussians, 
                gt_fmap, 
                sampled_index,
                render_visible_masks[viewpoint_cam.image_name],
            )

        loss = bce_loss(pred_heatmap, gt_heatmap)
        loss.backward()

        if iteration % grad_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        
        with torch.no_grad():
            loss_val = loss.item()
            
            if iteration % 10 == 0:
                if tb_writer:
                    tb_writer.add_scalar(
                    "detector_loss_patches/training_loss", loss_val, iteration
                )
                tb_writer.add_scalar(
                    "detector_loss_patches/lr",
                    optimizer.param_groups[0]["lr"],
                    iteration,
                )
        
        if (iteration + 1) in saving_iterations:
            print("\n[ITER {}] Saving detector".format(iteration + 1))
            torch.save(detector.state_dict(), save_path + f"/{iteration + 1}_detector.pth")

        if (iteration + 1) in testing_iterations:
            print("\n[ITER {}] Evaluating detector".format(iteration + 1))
            evaluate_detector(
                model_param=model_param,
                pipe_param=pipe_param,
                gaussians=gaussians,
                scene=scene,
                detector=detector,
                superpoint=model,
                sampled_idx=sampled_index,
                render_visible_masks=render_visible_masks,
                tb_writer=tb_writer,
                iteration=iteration,
            )

def train(
    args,
    model_param : ModelParams,
    opt_param : OptimizationParams, 
    pipe_param : PipelineParams,
    testing_iterations,
    saving_iterations,
):
    # Initialize the scene
    gaussians = GaussianModel(sh_degree=model_param.sh_degree)
    scene = Scene(
        model_param, 
        gaussians, 
        load_iteration=args.load_iteration, 
        shuffle=True,
        load_feature=True,
    )
    
    # Initialize the MLP
    # mlp = get_mlp_model(dim=16, type='SP').cuda()
    mlp = get_mlp_dataset(dim=16, dataset=args.mlp_method).eval().cuda()
    
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(log_dir=os.path.join(scene.model_path, pipe_param.checkpoint_dir, "detector"))
    
    training_detector(
        model_param=model_param,
        opt_param=opt_param, 
        pipe_param=pipe_param,
        gaussians=gaussians,
        scene=scene,
        mlp=mlp,
        testing_iterations=testing_iterations,
        saving_iterations=saving_iterations,
        tb_writer=tb_writer
    )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    Model_param = ModelParams(parser)
    Opt_param = OptimizationParams(parser)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--mlp_method", type=str)
    parser.add_argument("--load_iteration", type=int, default=30_000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # Initialize system state (RNG)
    print("\nTraining begins.")
    train(
        args=args,
        model_param=Model_param.extract(args),
        opt_param=Opt_param.extract(args), 
        pipe_param=Pipe_param.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
    )
    print("\nTraining complete.")