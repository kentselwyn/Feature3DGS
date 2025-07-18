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
import math
import torch
from tqdm import tqdm
from random import randint
from datetime import datetime
import torch.nn.functional as F
from utils.image_utils import psnr
###################################################
from gsplat import DefaultStrategy, rasterization
# from gaussian_renderer import render_gsplat
from gaussian_renderer import (
    create_gsplat_params, 
    create_gsplat_optimizers, 
    update_learning_rates, 
    render_with_gsplat,
    sync_gaussians_from_params
)
from scene import Scene
from scene.gaussian.gaussian_model import GaussianModel
###################################################
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from utils.loss_utils import l1_loss, ssim, l2_loss, weighted_l2
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(log_dir=args.model_path + f"/runs/{timestamp}")
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, Ll1_feature, Ll1_score, loss, l1_loss, elapsed, 
                    testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss_RGB', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_feature', Ll1_feature.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_score', Ll1_score.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : 
                               [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + \
                                    "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", renderArgs[0]["opacities"], iteration)
            tb_writer.add_scalar('total_points', renderArgs[0]["means"].shape[0], iteration)
        torch.cuda.empty_cache()


def training(model_param, opt_param, pipe_param, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(model_param)
    gaussians = GaussianModel(model_param.sh_degree)
    scene = Scene(model_param, gaussians, load_test_cams=hasattr(model_param, 'load_testcam') and model_param.load_testcam)

    # Initialize gsplat parameters and optimizers
    params = create_gsplat_params(gaussians)
    optimizers = create_gsplat_optimizers(params, opt_param)
    
    # Initialize the gsplat strategy
    strategy = DefaultStrategy(absgrad=model_param.use_abs_grad)
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state()

    # Get feature dimensions from sample camera
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    gt_feature_map = viewpoint_cam.semantic_feature.cuda()
    feature_out_dim = gt_feature_map.shape[0]
    gt_score_map = viewpoint_cam.score_feature.cuda()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt_param)
        # Re-create parameters after loading
        params = create_gsplat_params(gaussians)
        optimizers = create_gsplat_optimizers(params, opt_param)
        strategy_state = strategy.initialize_state()

    bg_color = [1, 1, 1] if model_param.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt_param.iterations), desc="Training progress")
    first_iter += 1

    torch.autograd.set_detect_anomaly(True)
    for iteration in range(first_iter, opt_param.iterations + 1):
        iter_start.record()
        
        # Update learning rates
        update_learning_rates(optimizers, iteration, opt_param)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render using gsplat
        if (iteration - 1) == debug_from:
            pipe_param.debug = True
            
        # Forward pass with gsplat
        render_colors, render_alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"], 
            scales=params["scales"],
            opacities=params["opacities"],
            colors=torch.cat([params["sh0"], params["shN"]], dim=1),
            viewmats=viewpoint_cam.world_view_transform.transpose(0, 1)[None].cuda(),
            Ks=torch.tensor([[
                viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx * 0.5)), 0, viewpoint_cam.image_width / 2.0,
                0, viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy * 0.5)), viewpoint_cam.image_height / 2.0,
                0, 0, 1
            ]], device="cuda").reshape(1, 3, 3),
            backgrounds=background[None],
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            packed=False,
            sh_degree=gaussians.active_sh_degree,
        )
        
        # Pre-backward step
        strategy.step_pre_backward(params, optimizers, strategy_state, iteration, info)
        
        # Extract rendered image and compute visibility
        image = render_colors[0].permute(2, 0, 1)
        radii = info["radii"].squeeze(0)
        radii_norm = torch.square(radii[:, 0]) + torch.square(radii[:, 1])
        visibility_filter = radii_norm > 0
        
        # Render feature and score maps
        visible_indices = visibility_filter
        if visible_indices.sum() > 0:
            # Feature map rendering
            loc_feature = params["loc_feature"][visible_indices]
            loc_feature = F.normalize(loc_feature, p=2, dim=-1)
            
            feat_colors, _, _ = rasterization(
                means=params["means"][visible_indices],
                quats=params["quats"][visible_indices],
                scales=params["scales"][visible_indices],
                opacities=params["opacities"][visible_indices],
                colors=loc_feature,
                viewmats=viewpoint_cam.world_view_transform.transpose(0, 1)[None].cuda(),
                Ks=torch.tensor([[
                    viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx * 0.5)), 0, viewpoint_cam.image_width / 2.0,
                    0, viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy * 0.5)), viewpoint_cam.image_height / 2.0,
                    0, 0, 1
                ]], device="cuda").reshape(1, 3, 3),
                width=viewpoint_cam.image_width,
                height=viewpoint_cam.image_height,
                packed=False,
            )
            feature_map = feat_colors[0].permute(2, 0, 1)
            feature_map = F.normalize(feature_map, p=2, dim=0)
            
            # Score map rendering
            score_feature = params["score_feature"][visible_indices]
            score_colors, _, _ = rasterization(
                means=params["means"][visible_indices],
                quats=params["quats"][visible_indices],
                scales=params["scales"][visible_indices], 
                opacities=params["opacities"][visible_indices],
                colors=score_feature,
                viewmats=viewpoint_cam.world_view_transform.transpose(0, 1)[None].cuda(),
                Ks=torch.tensor([[
                    viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx * 0.5)), 0, viewpoint_cam.image_width / 2.0,
                    0, viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy * 0.5)), viewpoint_cam.image_height / 2.0,
                    0, 0, 1
                ]], device="cuda").reshape(1, 3, 3),
                width=viewpoint_cam.image_width,
                height=viewpoint_cam.image_height,
                packed=False,
            )
            score_map = score_colors[0].permute(2, 0, 1)
        else:
            feature_map = torch.zeros_like(image)
            score_map = torch.zeros_like(image[:1])

        # Compute losses
        gt_image = viewpoint_cam.original_image.cuda().detach()
        gt_feature_map = viewpoint_cam.semantic_feature.cuda().detach()
        gt_score_map = viewpoint_cam.score_feature.cuda().detach()
        
        # Resize rendered maps to match ground truth
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), 
                                    mode='bilinear', align_corners=True).squeeze(0)
        score_map = F.interpolate(score_map.unsqueeze(0), size=(gt_score_map.shape[1], gt_score_map.shape[2]), 
                                  mode='bilinear', align_corners=True).squeeze(0)
        
        Ll1_feature = l1_loss(feature_map, gt_feature_map)
        
        # Score loss computation
        score_loss_comp = None
        if model_param.score_loss=="L2":
            score_loss_comp = l2_loss
        elif model_param.score_loss=="weighted":
            score_loss_comp = weighted_l2
        elif model_param.score_loss=="L1":
            score_loss_comp = l1_loss
        Ll1_score = score_loss_comp(score_map, gt_score_map)
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt_param.lambda_dssim) * Ll1 + opt_param.lambda_dssim * (1.0 - ssim(image, gt_image)) + \
               1.0 * Ll1_feature + model_param.score_scale * Ll1_score
        
        # Backward pass
        loss.backward()
        
        # Post-backward step
        strategy.step_post_backward(params, optimizers, strategy_state, iteration, info)
        
        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt_param.iterations:
                progress_bar.close()
                
            # Log and save
            def render_wrapper(viewpoint, params_dict, background):
                return render_with_gsplat(viewpoint, params_dict, background)
                
            training_report(tb_writer, iteration, Ll1, Ll1_feature, Ll1_score, loss, l1_loss, 
                            iter_start.elapsed_time(iter_end), testing_iterations, scene, render_wrapper, (params, background)) 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
            # Optimizer step (handled by strategy in post_backward)
            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # Convert gsplat params back to gaussians for saving
                sync_gaussians_from_params(gaussians, params)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    Model_param = ModelParams(parser)
    Opt_param = OptimizationParams(parser)
    Pipe_param = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(Model_param.extract(args), Opt_param.extract(args), Pipe_param.extract(args),
              args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")
