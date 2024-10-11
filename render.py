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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import matplotlib.pyplot as plt
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import render_path_spiral
import sklearn
import sklearn.decomposition
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from codes.used_codes.vis_scoremap import one_channel_vis
from models.networks import CNN_decoder


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



def interpolate_matrices(start_matrix, end_matrix, steps):
        # Generate interpolation factors
        interpolation_factors = np.linspace(0, 1, steps)
        # Interpolate between the matrices
        interpolated_matrices = []
        for factor in interpolation_factors:
            interpolated_matrix = (1 - factor) * start_matrix + factor * end_matrix
            interpolated_matrices.append(interpolated_matrix)
        return np.array(interpolated_matrices)



def multi_interpolate_matrices(matrix, num_interpolations):
    interpolated_matrices = []
    for i in range(matrix.shape[0] - 1):
        start_matrix = matrix[i]
        end_matrix = matrix[i + 1]
        for j in range(num_interpolations):
            t = (j + 1) / (num_interpolations + 1)
            interpolated_matrix = (1 - t) * start_matrix + t * end_matrix
            interpolated_matrices.append(interpolated_matrix)
    return np.array(interpolated_matrices)






def render_set(model_path, name, iteration, views, gaussians, pipe_param, background, speedup):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "image_renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "image_gt")
    
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_renders")
    gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_gt")
    saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_tensors")
    #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
    score_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "score_renders")
    gt_score_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "score_gt")
    saved_score_path = os.path.join(model_path, name, "ours_{}".format(iteration), "score_tensors")
    
    decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))
    
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth") ###
    
    if speedup:
        gt_feature_map = views[0].semantic_feature.cuda()
        feature_out_dim = gt_feature_map.shape[0]
        feature_in_dim = int(feature_out_dim/4)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    makedirs(feature_map_path, exist_ok=True)
    makedirs(gt_feature_map_path, exist_ok=True)
    makedirs(saved_feature_path, exist_ok=True)

    makedirs(score_map_path, exist_ok=True)
    makedirs(gt_score_map_path, exist_ok=True)
    makedirs(saved_score_path, exist_ok=True)


    # makedirs(depth_path, exist_ok=True) ###

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipe_param, background) 

        gt = view.original_image[0:3, :, :]
        gt_feature_map = view.semantic_feature.cuda() 
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # ############## depth
        # depth = render_pkg["depth"]
        # scale_nor = depth.max().item()
        # depth_nor = depth / scale_nor
        # depth_tensor_squeezed = depth_nor.squeeze()  # Remove the channel dimension
        # colormap = plt.get_cmap('jet')
        # depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
        # depth_colored_rgb = depth_colored[:, :, :3]
        # depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
        # output_path = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
        # depth_image.save(output_path)
        # ##############


        ############## visualize feature map
        feature_map = render_pkg["feature_map"][:16]

        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
        if speedup:
            feature_map = cnn_decoder(feature_map)

        feature_map_vis = feature_visualize_saving(feature_map)
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
        
        gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
        Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

        # save feature map
        feature_map = feature_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
        #############


        ############# score map
        score_map = render_pkg['score_map']

        gt_score_map = view.score_feature.cuda()
        score_map = F.interpolate(score_map.unsqueeze(0), size=(gt_score_map.shape[1], gt_score_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
        
        score_map_vis = one_channel_vis(score_map)
        score_map_vis.save(os.path.join(score_map_path, '{0:05d}'.format(idx) + "_score_vis.png"))
        
        gt_score_map_vis = one_channel_vis(gt_score_map)
        gt_score_map_vis.save(os.path.join(gt_score_map_path, '{0:05d}'.format(idx) + "_score_vis.png"))
        
        # save feature map
        score_map = score_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(score_map).half(), os.path.join(saved_score_path, '{0:05d}'.format(idx) + "_smap_CxHxW.pt"))
        #############



###
def render_novel_views(model_path, name, iteration, views, gaussians, pipe_param, background, 
                       speedup, multi_interpolate, num_views):
    if multi_interpolate:
        name = name + "_multi_interpolate"
    # make dirs
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "image_renders")
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_renders")
    saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_tensors")
    #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
    decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))
    
    score_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "score_renders")
    saved_score_path = os.path.join(model_path, name, "ours_{}".format(iteration), "score_tensors")

    if speedup:
        gt_feature_map = views[0].semantic_feature.cuda()
        feature_out_dim = gt_feature_map.shape[0]
        feature_in_dim = int(feature_out_dim/4)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))
    
    makedirs(render_path, exist_ok=True)
    makedirs(feature_map_path, exist_ok=True)
    makedirs(saved_feature_path, exist_ok=True)

    makedirs(score_map_path, exist_ok=True)
    makedirs(saved_score_path, exist_ok=True)

    view = views[0]
    
    # create novel poses
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose) 
    if not multi_interpolate:
        poses = interpolate_matrices(render_poses[0], render_poses[-1], num_views)
    else:
        poses = multi_interpolate_matrices(np.array(render_poses), 2)

    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        render_pkg = render(view, gaussians, pipe_param, background) 

        
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
        
        
        ########## visualize feature map
        gt_feature_map = view.semantic_feature.cuda() 
        feature_map = render_pkg["feature_map"]
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
        if speedup:
            feature_map = cnn_decoder(feature_map)

        feature_map_vis = feature_visualize_saving(feature_map)
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

        # save feature map
        feature_map = feature_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
        ##########


        ########## visualize score map
        gt_score_map = view.score_feature.cuda()
        score_map = render_pkg['score_map']
        score_map = F.interpolate(score_map.unsqueeze(0), size=(gt_score_map.shape[1], gt_score_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
        score_map_vis = one_channel_vis(score_map)
        score_map_vis.save(os.path.join(score_map_path, '{0:05d}'.format(idx) + "_score_vis.png"))
        
        score_map = score_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(score_map).half(), os.path.join(saved_score_path, '{0:05d}'.format(idx) + "_smap_CxHxW.pt"))




def render_sets(model_param : ModelParams, iteration : int, pipe_param : PipelineParams, skip_train : bool, skip_test : bool, 
                novel_view : bool, multi_interpolate : bool, num_views : int): 
    with torch.no_grad():
        gaussians = GaussianModel(model_param.sh_degree)
        scene = Scene(model_param, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if model_param.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(model_param.model_path, "rendering/trains", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe_param, background, model_param.speedup)

        if not skip_test:
             render_set(model_param.model_path, "rendering/test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe_param, background, model_param.speedup)

        if novel_view:
             render_novel_views(model_param.model_path, "rendering/novel_views", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe_param, background, model_param.speedup, 
                                multi_interpolate, num_views)




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--novel_view", action="store_true") ###
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--multi_interpolate", action="store_true") ###
    parser.add_argument("--num_views", default=100, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(Model_param.extract(args), args.iteration, Pipe_param.extract(args), 
                args.skip_train, args.skip_test, args.novel_view,
                args.multi_interpolate, args.num_views)
    
    