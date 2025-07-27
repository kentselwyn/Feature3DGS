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
import sys
import uuid
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from random import randint
from datetime import datetime
from argparse import ArgumentParser, Namespace

from utils.image_utils import psnr
from utils.general_utils import safe_state, seed_everything
from utils.loss_utils import l1_loss, ssim, l2_loss, weighted_l2
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render_gsplat
from scene import Scene
from scene.gaussian.gaussian_model import GaussianModel

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):
    """Set up output directory and tensorboard logging"""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, Ll1_feature, Ll1_score, loss, elapsed,
                   testing_iterations, scene: Scene, background, model_param,
                   render_func=None, masks=None):
    """Report training progress and validation metrics"""
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss_RGB', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_feature', Ll1_feature.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_score', Ll1_score.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()}, 
            {'name': 'train', 'cameras': 
             [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                l1_feature_test = 0.0
                l1_score_test = 0.0
                psnr_test = 0.0
                
                for idx, viewpoint in enumerate(config['cameras']):
                    if render_func:
                        render_pkg = render_func(viewpoint, scene.gaussians, background)
                    else:
                        render_pkg = render_gsplat(viewpoint, scene.gaussians, background, rgb_only=False)
                    
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    feature_map = render_pkg.get("feature_map", None)
                    score_map = render_pkg.get("score_map", None)
                    
                    # Get ground truth
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_feature_map = viewpoint.semantic_feature.cuda()
                    gt_score_map = viewpoint.score_feature.cuda()
                    
                    # Apply masks if available
                    if masks is not None and viewpoint.image_name in masks:
                        mask_data = masks[viewpoint.image_name]
                        if len(mask_data) >= 3:
                            obj_mask = mask_data[0].cuda()[None]
                            sky_mask = mask_data[1].cuda()[None] 
                            distort_mask = mask_data[2].cuda()[None]
                            mask = obj_mask & distort_mask
                            
                            image = image * mask
                            gt_image = gt_image * mask
                            gt_image[sky_mask.repeat(3, 1, 1) == False] = 1
                    
                    # Compute losses
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    if feature_map is not None and gt_feature_map is not None:
                        # Resize feature map to match GT if needed
                        if feature_map.shape != gt_feature_map.shape:
                            feature_map = F.interpolate(
                                feature_map.unsqueeze(0), 
                                size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                                mode='bilinear', align_corners=True
                            ).squeeze(0)
                        l1_feature_test += l1_loss(feature_map, gt_feature_map).mean().double()
                    
                    if score_map is not None and gt_score_map is not None:
                        # Resize score map to match GT if needed
                        if score_map.shape != gt_score_map.shape:
                            score_map = F.interpolate(
                                score_map.unsqueeze(0),
                                size=(gt_score_map.shape[1], gt_score_map.shape[2]), 
                                mode='bilinear', align_corners=True
                            ).squeeze(0)
                        l1_score_test += l1_loss(score_map, gt_score_map).mean().double()
                    
                    # Log images to tensorboard
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                            image[None], global_step=iteration
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                                gt_image[None], global_step=iteration
                            )

                # Average metrics
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                l1_feature_test /= len(config['cameras'])
                l1_score_test /= len(config['cameras'])
                
                print("\n[ITER {}] Evaluating {}: L1 {:.6f} PSNR {:.2f} Feature L1 {:.6f} Score L1 {:.6f}".format(
                    iteration, config['name'], l1_test, psnr_test, l1_feature_test, l1_score_test
                ))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_feature', l1_feature_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_score', l1_score_test, iteration)
        
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        
        torch.cuda.empty_cache()


def training(model_param, opt_param, pipe_param, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from):
    """Main training function"""
    print("Starting Feature3DGS training...")
    print(f"Model parameters: {model_param}")
    print(f"Optimization parameters: {opt_param}")
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(model_param)
    
    # Initialize Gaussian model and scene
    gaussians = GaussianModel(model_param.sh_degree)
    scene = Scene(model_param, gaussians, 
                  load_test_cams=hasattr(model_param, 'load_testcam') and model_param.load_testcam)

    # Load masks if available
    masks = None
    if hasattr(model_param, 'source_path'):
        mask_path = os.path.join(model_param.source_path, "masks.pkl")
        if os.path.exists(mask_path):
            print(f"Loading masks from {mask_path}")
            with open(mask_path, 'rb') as f:
                masks = pickle.load(f)

    # Setup training
    gaussians.training_setup(opt_param)
    
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt_param)

    # Background color
    bg_color = [1, 1, 1] if model_param.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Training setup
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt_param.iterations), desc="Feature3DGS Training")
    first_iter += 1

    # Training loop
    for iteration in range(first_iter, opt_param.iterations + 1):
        iter_start.record()
        
        # Update learning rates
        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            
        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Debug mode
        if (iteration - 1) == debug_from:
            pipe_param.debug = True
            
        # Render the scene
        render_pkg = render_gsplat(viewpoint_cam, gaussians, background, rgb_only=False)
        
        image = render_pkg["render"]
        feature_map = render_pkg.get("feature_map", None)
        score_map = render_pkg.get("score_map", None) 
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        # Get ground truth data
        gt_image = viewpoint_cam.original_image.cuda()
        gt_feature_map = viewpoint_cam.semantic_feature.cuda()
        gt_score_map = viewpoint_cam.score_feature.cuda()
        
        # Apply masks if available
        if masks is not None and viewpoint_cam.image_name in masks:
            mask_data = masks[viewpoint_cam.image_name]
            if len(mask_data) >= 3:
                obj_mask = mask_data[0].cuda()[None]
                sky_mask = mask_data[1].cuda()[None]
                distort_mask = mask_data[2].cuda()[None]
                mask = obj_mask & distort_mask
                
                image = image * mask
                gt_image = gt_image * mask
                gt_image[sky_mask.repeat(3, 1, 1) == False] = 1
        
        # Compute RGB loss
        Ll1 = l1_loss(image, gt_image)
        
        # Compute feature loss
        if feature_map is not None and gt_feature_map is not None:
            # Resize feature map to match ground truth if needed
            if feature_map.shape != gt_feature_map.shape:
                feature_map = F.interpolate(
                    feature_map.unsqueeze(0), 
                    size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                    mode='bilinear', align_corners=True
                ).squeeze(0)
            Ll1_feature = l1_loss(feature_map, gt_feature_map)
        else:
            Ll1_feature = torch.tensor(0.0, device="cuda")
        
        # Compute score loss
        if score_map is not None and gt_score_map is not None:
            # Resize score map to match ground truth if needed
            if score_map.shape != gt_score_map.shape:
                score_map = F.interpolate(
                    score_map.unsqueeze(0),
                    size=(gt_score_map.shape[1], gt_score_map.shape[2]),
                    mode='bilinear', align_corners=True
                ).squeeze(0)
            
            # Use the specified score loss function
            score_loss_func = l1_loss  # default
            if hasattr(model_param, 'score_loss'):
                if model_param.score_loss == "L2":
                    score_loss_func = l2_loss
                elif model_param.score_loss == "weighted":
                    score_loss_func = weighted_l2
                elif model_param.score_loss == "L1":
                    score_loss_func = l1_loss
            
            Ll1_score = score_loss_func(score_map, gt_score_map)
        else:
            Ll1_score = torch.tensor(0.0, device="cuda")
        
        # Total loss
        loss = ((1.0 - opt_param.lambda_dssim) * Ll1 + 
                opt_param.lambda_dssim * (1.0 - ssim(image, gt_image)) +
                1.0 * Ll1_feature + 
                getattr(model_param, 'score_scale', 1.0) * Ll1_score)
        
        # Backward pass
        loss.backward()
        
        iter_end.record()

        # viewspace_point_tensor has shape [1, N, 2], need to squeeze to [N, 2]
        if viewspace_point_tensor.dim() == 3:
            viewspace_points_2d = viewspace_point_tensor.squeeze(0)
        else:
            viewspace_points_2d = viewspace_point_tensor
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt_param.iterations:
                progress_bar.close()
                
            # Training report
            training_report(tb_writer, iteration, Ll1, Ll1_feature, Ll1_score, loss,
                          iter_start.elapsed_time(iter_end), testing_iterations, scene, 
                          background, model_param, masks=masks)
            
            # Save gaussians
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                
            # Densification
            if iteration < opt_param.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                
                # Add densification stats
                # gaussians.add_densification_stats(viewspace_points_2d, visibility_filter)
                gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    visibility_filter,
                    image.shape[2],
                    image.shape[1],
                )

                if (iteration > opt_param.densify_from_iter and 
                    iteration % opt_param.densification_interval == 0):
                    size_threshold = 20 if iteration > opt_param.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt_param.densify_grad_threshold, 0.005, 
                        scene.cameras_extent, size_threshold
                    )

                if (iteration % opt_param.opacity_reset_interval == 0 or 
                    (model_param.white_background and iteration == opt_param.densify_from_iter)):
                    gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < opt_param.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
            # Save checkpoint
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), 
                          scene.model_path + f"/chkpnt{iteration}.pth")

    # Final save
    print("\nTraining complete. Saving final model...")
    scene.save(opt_param.iterations)
    
    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    # Set random seed for reproducibility
    seed_everything(2025)
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Feature3DGS Training script parameters")
    model_params = ModelParams(parser)
    opt_params = OptimizationParams(parser)
    pipe_params = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, 
                       default=[7_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, 
                       default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    print(f"Command line arguments: {args}")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # Enable anomaly detection if requested
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # Start training
    training(model_params.extract(args), opt_params.extract(args), pipe_params.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
             args.start_checkpoint, args.debug_from)
    
    print("\nTraining complete!")
