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
import torch
from tqdm import tqdm
from random import randint
from datetime import datetime
import torch.nn.functional as F
from utils.image_utils import psnr
###################################################
from gaussian_renderer import render
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
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
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
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def training(model_param, opt_param, pipe_param, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(model_param)
    gaussians = GaussianModel(model_param.sh_degree)
    scene = Scene(model_param, gaussians, load_testcam=args.load_testcam)

    # 2D semantic feature map CNN decoder
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    gt_feature_map = viewpoint_cam.semantic_feature.cuda()
    feature_out_dim = gt_feature_map.shape[0]
    gt_score_map = viewpoint_cam.score_feature.cuda()

    gaussians.training_setup(opt_param)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt_param)

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
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe_param.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe_param, background)
        feature_map, score_map, image, viewspace_point_tensor, visibility_filter, radii = render_pkg["feature_map"], \
                    render_pkg["score_map"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
                    render_pkg["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda().detach()
        gt_feature_map = viewpoint_cam.semantic_feature.cuda().detach()
        # ### score
        gt_score_map = viewpoint_cam.score_feature.cuda().detach()
        gt_score_map = gt_score_map.cuda().detach()
    
        # adjust_gt_feature = F.interpolate(gt_image.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), 
                                    mode='bilinear', align_corners=True).squeeze(0)
        score_map = F.interpolate(score_map.unsqueeze(0), size=(gt_score_map.shape[1], gt_score_map.shape[2]), 
                                  mode='bilinear', align_corners=True).squeeze(0)
        Ll1_feature = l1_loss(feature_map, gt_feature_map)
        score_loss_comp = None
        if model_param.score_loss=="L2":
            score_loss_comp = l2_loss
        elif model_param.score_loss=="weighted":
            score_loss_comp = weighted_l2
        elif model_param.score_loss=="L1":
            score_loss_comp = l1_loss
        Ll1_score = score_loss_comp(score_map, gt_score_map)
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt_param.lambda_dssim) * Ll1 + opt_param.lambda_dssim * (1.0 - ssim(image, gt_image)) +  1.0 * Ll1_feature + \
                model_param.score_scale* Ll1_score
        loss.backward()
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
            training_report(tb_writer, iteration, Ll1, Ll1_feature, Ll1_score, loss, l1_loss, 
                            iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe_param, background)) 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
            # Densification
            if iteration < opt_param.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt_param.densify_from_iter and iteration % opt_param.densification_interval == 0:
                    size_threshold = 20 if iteration > opt_param.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt_param.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt_param.opacity_reset_interval == 0 or (model_param.white_background and iteration == opt_param.densify_from_iter):
                    gaussians.reset_opacity()
            # Optimizer step
            if iteration < opt_param.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
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
