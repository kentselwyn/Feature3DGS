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
import torch
import sklearn
import torchvision
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from os import makedirs
from scene import Scene
from pathlib import Path
import sklearn.decomposition
import torch.nn.functional as F
from matchers.lightglue import LightGlue
from argparse import ArgumentParser
from gaussian_renderer import render
from copy import deepcopy
from encoders.superpoint.superpoint import SuperPoint
from utils.general_utils import PILtoTorch
from utils.general_utils import safe_state
from scene.gaussian.gaussian_model import GaussianModel
from utils.graphics_utils import getWorld2View2
from utils.scoremap_vis import one_channel_vis
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.loc.loc_utils import choose_th
from utils.match.match_img import find_small_circle_centers, sample_descriptors_fix_sampling, \
                                    match_data, save_matchimg_th
from utils.match.metrics_match import compute_metrics
from detector.models import RefinedScoreNet
import copy

# utils
def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, N, h, w])
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


def match_and_save(kpts0, kpts1, desc0, desc1, img0, img1,
                   imgsize, poses, match_path, matcher):
    data = {}
    data["keypoints0"] = kpts0
    data["keypoints1"] = kpts1
    data["descriptors0"] = desc0
    data["descriptors1"] = desc1
    data["image_size"] = imgsize
    result = match_data(data, matcher, 
                        img0=img0,
                        img1=img1)
    result.update(poses)
    compute_metrics(result)
    log_path = f"{str(Path(match_path).parent.parent)}/log.txt"
    threshold = args.match_precision_th
    epi_good = result['epi_errs'][0] < threshold
    precision = (epi_good.sum()/len(epi_good))*100
    R_err = result['R_errs'][0]
    t_err = result['t_errs'][0]
    match_num = len(result['mkpt0'])
    name = result['identifiers'][0]
    with open(log_path, 'a') as f:  # 'a' for append mode
        f.write(f"Name: {name}\n")
        f.write(f"MatchNum: {match_num},   Precision: {precision:.2f}%\n")
        # f.write(f"Precision     : {precision:.2f}%\n")
        f.write(f"Rotation Error: {R_err:.4f}\n")
        f.write(f"Transl. Error : {t_err:.4f}\n")
        f.write(f"{'-'*40}\n\n")
    save_matchimg_th(result, match_path, threshold=threshold)



# render train, test
def render_set(model_path, name, iteration, views, gaussians, pipe_param, background, 
               args, mlp, render_encoder):
    ###################################################
    myHist = args.hist
    detector_kernel_size = 7
    detector_hist = 0.9
    ###################################################
    fin_name = f"ours_{iteration}_{len(views)}"
    desc_path = os.path.join(model_path, name, fin_name, "desc")
    mat_path =  os.path.join(model_path, name, fin_name, 
                             f"match_{detector_kernel_size}_{detector_hist}_{args.match_precision_th}")
    ###################################################
    RenderRe_path = os.path.join(mat_path, "RenderRender")
    RenderGT_path = os.path.join(mat_path, "RenderGT")
    GT_USE_FT_for_kpt = os.path.join(RenderGT_path, "kpt_from_FT")
    GT_USE_SP_for_kpt = os.path.join(RenderGT_path, "kpt_from_SP")
    GT_USE_DE_for_kpt = os.path.join(RenderGT_path, "kpt_from_DE")
    
    render_path = os.path.join(desc_path, "image", "renders")
    gts_path = os.path.join(desc_path, "image", "gt")
    
    feature_map_path = os.path.join(desc_path, "feature", "renders")
    gt_feature_map_path = os.path.join(desc_path, "feature", "gt")
    
    score_map_path = os.path.join(desc_path, "score", "renders")
    gt_score_map_path = os.path.join(desc_path, "score", "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(feature_map_path, exist_ok=True)
    makedirs(gt_feature_map_path, exist_ok=True)
    makedirs(score_map_path, exist_ok=True)
    makedirs(gt_score_map_path, exist_ok=True)
    if args.detector_path is not None:
        model_path = args.detector_path
        encoder = copy.deepcopy(render_encoder)
        detector = RefinedScoreNet(encoder, False)
        ckpt = torch.load(model_path)
        detector.load_state_dict(ckpt)
        detector = detector.cuda().eval()
        for param in detector.parameters():
            param.requires_grad = False
        grad_params = [p for p in detector.parameters() if p.requires_grad]
    if args.render_kpt_desc:
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            print(view.image_name)
            render_pkg = render(view, gaussians, pipe_param, background)
            gt_img = view.original_image[0:3, :, :]
            gt_feature_map = view.semantic_feature.cuda() 
            torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, f"{view.image_name}.png")) 
            torchvision.utils.save_image(gt_img, os.path.join(gts_path, f"{view.image_name}.png"))
            ############## visualize feature map
            feature_map = render_pkg["feature_map"][:16]
            feature_map = F.interpolate(feature_map.unsqueeze(0), 
                                        size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), 
                                                mode='bilinear', align_corners=True).squeeze(0) ###
            feature_map_vis = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(feature_map_path, f"{view.image_name}.png"))
            
            gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
            Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(gt_feature_map_path, f"{view.image_name}.png"))
            #############
            ############# score map
            score_map = render_pkg['score_map']
            gt_score_map = view.score_feature.cuda()
            score_map = F.interpolate(score_map.unsqueeze(0), 
                                    size=(gt_score_map.shape[1], gt_score_map.shape[2]), 
                                                mode='bilinear', align_corners=True).squeeze(0) ###
            score_map_vis = one_channel_vis(score_map)
            score_map_vis.save(os.path.join(score_map_path, f"{view.image_name}.png"))
            gt_score_map_vis = one_channel_vis(gt_score_map)
            gt_score_map_vis.save(os.path.join(gt_score_map_path, f"{view.image_name}.png"))
            #############
            ############# detector score map
            #############
    if args.render_match:
        matcher = LightGlue({"filter_threshold": args.lg_th ,}).cuda().eval()
        ########################################## RenderRe ##########################################
        match_path_0 = os.path.join(RenderRe_path, f"(Ft)(Ft)-(Ft)(Ft)",    f"{args.lg_th}_{myHist}", "imgs")
        match_path_1 = os.path.join(RenderRe_path, f"(Ft)(Ft)-(DE)(Ft)",    f"{args.lg_th}_{myHist}", "imgs")
        match_path_11= os.path.join(RenderRe_path, f"(Ft)(Ft)-(SP)(Ft)",    f"{args.lg_th}_{myHist}_{args.sp_kpt}_{args.sp_th}", "imgs")
        match_path_12= os.path.join(RenderRe_path, f"(SP)(Ft)-(SP)(Ft)",    f"{args.lg_th}_{args.sp_kpt}_{args.sp_th}", "imgs")
        match_path_2 = os.path.join(RenderRe_path, f"(SP)(SP)-(SP)(SP)",    f"{args.lg_th}_{args.sp_kpt}_{args.sp_th}", "imgs")
        match_path_13= os.path.join(RenderRe_path, f"(Ft)(SP)-(Ft)(SP)",    f"{args.lg_th}_{myHist}", "imgs")
        ########################################## RenderGT ##########################################
        ################ SP
        match_path_3 = os.path.join(GT_USE_SP_for_kpt, f"(Ft)(Ft)-(SP)(SP)",    f"{args.lg_th}_{myHist}_{args.sp_kpt}_{args.sp_th}", "imgs")
        match_path_4 = os.path.join(GT_USE_SP_for_kpt, f"(Ft)(Ft)-(SP)(SPMLP)", f"{args.lg_th}_{myHist}_{args.sp_kpt}_{args.sp_th}", "imgs")
        match_path_5 = os.path.join(GT_USE_SP_for_kpt, f"(SP)(SP)-(SP)(SP)",    f"{args.lg_th}_{args.sp_kpt}_{args.sp_th}", "imgs")
        match_path_6 = os.path.join(GT_USE_SP_for_kpt, f"(Ft)(SP)-(SP)(SP)",    f"{args.lg_th}_{myHist}_{args.sp_kpt}_{args.sp_th}", "imgs")
        ################ Ft
        match_path_7 = os.path.join(GT_USE_FT_for_kpt, f"(Ft)(Ft)-(Ft)(SP)",    f"{args.lg_th}_{myHist}", "imgs")
        match_path_8 = os.path.join(GT_USE_FT_for_kpt, f"(Ft)(SP)-(Ft)(SP)",    f"{args.lg_th}_{myHist}", "imgs")
        ################ DE
        match_path_9 = os.path.join(GT_USE_DE_for_kpt, f"(Ft)(SP)-(DE)(SP)",    f"{args.lg_th}_{myHist}", "imgs")
        match_path_10= os.path.join(GT_USE_DE_for_kpt, f"(Ft)(Ft)-(DE)(SP)",    f"{args.lg_th}_{myHist}", "imgs")
        ##############################################################################################
        makedirs(match_path_0, exist_ok=True)
        makedirs(match_path_1, exist_ok=True)
        makedirs(match_path_11, exist_ok=True)
        makedirs(match_path_2, exist_ok=True)
        makedirs(match_path_12, exist_ok=True)
        makedirs(match_path_13, exist_ok=True)
        ################ SP
        makedirs(match_path_3, exist_ok=True)
        makedirs(match_path_4, exist_ok=True)
        makedirs(match_path_5, exist_ok=True)
        makedirs(match_path_6, exist_ok=True)
        ################ Ft
        makedirs(match_path_7, exist_ok=True)
        makedirs(match_path_8, exist_ok=True)
        ################ DE
        makedirs(match_path_9, exist_ok=True)
        makedirs(match_path_10, exist_ok=True)
        
        pkg0 = render(views[0], gaussians, Pipe_param.extract(args), background)

        K = torch.tensor((views[0].intrinsic_matrix).astype(np.float32))
        T0 = views[0].extrinsic_matrix
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if idx==0:
                continue
            pkg1 = render(view, gaussians, Pipe_param.extract(args), background)
            T1 = view.extrinsic_matrix
            s0, s1, f0, f1 = pkg0['score_map'], pkg1['score_map'], \
                                    pkg0['feature_map'], pkg1['feature_map']
            th0 = choose_th(s0, myHist)
            th1 = choose_th(s1, myHist)
            kpt0 = find_small_circle_centers(s0, threshold=th0, kernel_size=15).clone().detach()[:, [1, 0]]
            kpt1 = find_small_circle_centers(s1, threshold=th1, kernel_size=15).clone().detach()[:, [1, 0]]
            _, h, w = s0.shape
            scale = torch.tensor([w, h]).to(s0)
            feat0 = sample_descriptors_fix_sampling(kpt0, f0, scale)
            feat1 = sample_descriptors_fix_sampling(kpt1, f1, scale)
            feat0 = mlp.decode(feat0)
            feat1 = mlp.decode(feat1)
            ##############################################
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)
            T_1to0 = T_0to1.inverse()
            pose_data = {
                "K0": K,
                "K1": K,
                "T_0to1": T_0to1,
                "T_1to0": T_1to0,
                "identifiers": [f"{view.image_name}"],
            }
            img_render0 = pkg0['render']
            img_render1 = pkg1['render']
            img_gt1 = view.original_image
            
            # 用render kpt抽取superpoint desc, imgRender0
            tmp = {}
            tmp["image"] = img_render0.unsqueeze(0).cuda()
            SP_pred0 = render_encoder(tmp)
            feat_sp_dense0 = SP_pred0["dense_descriptors"].squeeze(0)
            feat_sp0 = sample_descriptors_fix_sampling(kpt0, feat_sp_dense0, scale)
            
            # 單純測試
            # 用render kpt抽取superpoint desc, imgGT1, 不會出現
            tmp = {}
            tmp["image"] = img_gt1.unsqueeze(0).cuda()
            SP_pred_gt1 = render_encoder(tmp)
            feat_sp_gt_dense1 = SP_pred_gt1["dense_descriptors"].squeeze(0)
            feat_sp1_gt = sample_descriptors_fix_sampling(kpt1, feat_sp_gt_dense1, scale)

            # detection part
            tmp = {}
            tmp["image"] = img_gt1.unsqueeze(0).cuda()
            detect_gt_map1 = detector(tmp["image"])[0]
            th_detect = choose_th(detect_gt_map1, detector_hist)
            detect_kpt_gt1 = find_small_circle_centers(detect_gt_map1, threshold=th_detect, kernel_size=detector_kernel_size).clone().detach()[:, [1, 0]]
            # 用detect kpt抽取 superpoint desc1
            feat_sp1_gt_detect = sample_descriptors_fix_sampling(detect_kpt_gt1, feat_sp_gt_dense1, scale)
            # 用detect kpt抽取 render desc1
            feat1_detect =       sample_descriptors_fix_sampling(detect_kpt_gt1, f1,                scale)
            feat1_detect = mlp.decode(feat1_detect)
            # 用SP kpt抽取 render desc1
            featttttt1 = sample_descriptors_fix_sampling(SP_pred_gt1["keypoints"][0], f1,           scale)
            featttttt1 = mlp.decode(featttttt1)
            # 用SP kpt抽取 render desc0
            featttttt0 = sample_descriptors_fix_sampling(SP_pred0["keypoints"][0], f0,           scale)
            featttttt0 = mlp.decode(featttttt0)
            ########################################################################################### RenderRe ##########################################
            ####################### 兩邊都render, 不會出現 ####################
            #### (Ft)(Ft)-(Ft)(Ft) 看render情況下ours
            match_and_save(kpt0.unsqueeze(0),              kpt1.unsqueeze(0), 
                           feat0,                          feat1,
                           img_render0.permute(1, 2, 0),   img_render1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,        f"{match_path_0}/{view.image_name}.png",
                           matcher)
            #### (Ft)(Ft)-(DE)(Ft) 看使用 detector+render feature 的match情況
            match_and_save(kpt0.unsqueeze(0),             detect_kpt_gt1.unsqueeze(0), 
                           feat0,                         feat1_detect,
                           img_render0.permute(1, 2, 0),  img_render1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_1}/{view.image_name}.png",
                           matcher)
            #### (Ft)(Ft)-(SP)(Ft) 看使用 SP kpt+render feature 的match情況
            match_and_save(kpt0.unsqueeze(0),             SP_pred_gt1["keypoints"], 
                           feat0,                         featttttt1,
                           img_render0.permute(1, 2, 0),  img_render1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_11}/{view.image_name}.png",
                           matcher)
            #### (SP)(Ft)-(SP)(Ft) 兩邊都是 SP kpt+render feature 的match情況
            match_and_save(SP_pred0["keypoints"],         SP_pred_gt1["keypoints"], 
                           featttttt0,                    featttttt1,
                           img_render0.permute(1, 2, 0),  img_render1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_12}/{view.image_name}.png",
                           matcher)
            #### (SP)(SP)-(SP)(SP) 看render情況下rival的情況
            tmp = {}
            tmp["image"] = img_render1.unsqueeze(0).cuda()
            SP_pred1 = render_encoder(tmp)
            match_and_save(SP_pred0["keypoints"],          SP_pred1["keypoints"], 
                           SP_pred0["descriptors"],        SP_pred1["descriptors"],
                           img_render0.permute(1, 2, 0),   img_render1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,        f"{match_path_2}/{view.image_name}.png",
                           matcher)
            #### (Ft)(SP)-(Ft)(SP) 用ft keyppoint抽SP desc
            feat_sp_dense1 = SP_pred1["dense_descriptors"].squeeze(0)
            feat_sp1 = sample_descriptors_fix_sampling(kpt1, feat_sp_dense1, scale)
            match_and_save(kpt0.unsqueeze(0),              kpt1.unsqueeze(0), 
                           feat_sp0,                       feat_sp1,
                           img_render0.permute(1, 2, 0),   img_render1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,        f"{match_path_13}/{view.image_name}.png",
                           matcher)
            ########################################################################################### RenderGT ##########################################
            ####################### 用SP抽GT keypoints 真實情況 #####################
            #### (Ft)(Ft)-(SP)(SP) ours目前localizae方法
            match_and_save(kpt0.unsqueeze(0),             SP_pred_gt1["keypoints"], 
                           feat0,                         SP_pred_gt1["descriptors"],
                           img_render0.permute(1, 2, 0),  img_gt1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_3}/{view.image_name}.png",
                           matcher)
            #### (Ft)(Ft)-(SP)(SPMLP) ours目前localizae方法+MLP
            match_and_save(kpt0.unsqueeze(0),             SP_pred_gt1["keypoints"], 
                           feat0,                         mlp.decode(mlp(SP_pred_gt1["descriptors"])),
                           img_render0.permute(1, 2, 0),  img_gt1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_4}/{view.image_name}.png",
                           matcher)
            #### (SP)(SP)-(SP)(SP)  rival目前localize方法
            match_and_save(SP_pred0["keypoints"],         SP_pred_gt1["keypoints"], 
                           SP_pred0["descriptors"],       SP_pred_gt1["descriptors"],
                           img_render0.permute(1, 2, 0),  img_gt1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_5}/{view.image_name}.png",
                           matcher)
            #### (Ft)(SP)-(SP)(SP)  ours目前localizae方法, feature從SP抽取
            match_and_save(kpt0.unsqueeze(0),             SP_pred_gt1["keypoints"], 
                           feat_sp0,                      SP_pred_gt1["descriptors"],
                           img_render0.permute(1, 2, 0),  img_gt1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_6}/{view.image_name}.png",
                           matcher)
            ####################### 用Ft抽GT keypoints 不會出現 單純測試 ###############
            #### (Ft)(Ft)-(Ft)(SP) 使用render的keypoints抽取ground truth SP的desc, 效果不錯, 一邊用Ft, 一邊用SP
            match_and_save(kpt0.unsqueeze(0),             kpt1.unsqueeze(0), 
                           feat0,                         feat_sp1_gt,
                           img_render0.permute(1, 2, 0),  img_gt1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_7}/{view.image_name}.png",
                           matcher)
            #### (Ft)(SP)-(Ft)(SP)  
            match_and_save(kpt0.unsqueeze(0),             kpt1.unsqueeze(0), 
                           feat_sp0,                      feat_sp1_gt,
                           img_render0.permute(1, 2, 0),  img_gt1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_8}/{view.image_name}.png",
                           matcher)
            ####################### 用DE抽GT keypoints, 真實情況, 看detector效果 #######
            #### (Ft)(SP)-(DE)(SP) 兩邊都用SP feature
            match_and_save(kpt0.unsqueeze(0),             detect_kpt_gt1.unsqueeze(0), 
                           feat_sp0,                      feat_sp1_gt_detect,
                           img_render0.permute(1, 2, 0),  img_gt1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_9}/{view.image_name}.png",
                           matcher)
            #### (Ft)(Ft)-(DE)(SP) render的用Ft
            match_and_save(kpt0.unsqueeze(0),             detect_kpt_gt1.unsqueeze(0), 
                           feat0,                         feat_sp1_gt_detect,
                           img_render0.permute(1, 2, 0),  img_gt1.permute(1, 2, 0),
                           s0.shape[1:], pose_data,       f"{match_path_10}/{view.image_name}.png",
                           matcher)
            ##########################################
            pkg0 = deepcopy(pkg1)
            T0 = deepcopy(T1)



# render novel view
def interpolate_matrices(start_matrix, end_matrix, steps):
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

def render_novel_views(model_path, name, iteration, views, gaussians, pipe_param, background, 
                       speedup, multi_interpolate):
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
        poses = interpolate_matrices(render_poses[0], render_poses[-1], 10)
    else:
        poses = multi_interpolate_matrices(np.array(render_poses), 2)
    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        render_pkg = render(view, gaussians, pipe_param, background)
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, f"{view.image_name}.png")) 
        ########## visualize feature map
        gt_feature_map = view.semantic_feature.cuda()
        feature_map = render_pkg["feature_map"]
        feature_map = F.interpolate(feature_map.unsqueeze(0), 
                                    size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), 
                                            mode='bilinear', align_corners=True).squeeze(0) ###
        feature_map_vis = feature_visualize_saving(feature_map)
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8))\
            .save(os.path.join(feature_map_path,view.image_name + "_feature_vis.png"))
        feature_map = feature_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, view.image_name + "_fmap_CxHxW.pt"))
        ########## visualize score map
        gt_score_map = view.score_feature.cuda()
        score_map = render_pkg['score_map']
        score_map = F.interpolate(score_map.unsqueeze(0), size=(gt_score_map.shape[1], 
                                                                gt_score_map.shape[2]), mode='bilinear', 
                                                                align_corners=True).squeeze(0) ###
        score_map_vis = one_channel_vis(score_map)
        score_map_vis.save(os.path.join(score_map_path, view.image_name + "_score_vis.png"))
        score_map = score_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(score_map).half(), os.path.join(saved_score_path, view.image_name + "_smap_CxHxW.pt"))



# render pairs
def c2w_to_w2c(T_cw):
    R = T_cw[:3, :3]  # Top-left 3x3 part is the rotation matrix
    t = T_cw[:3, 3]   # Top-right 3x1 part is the translation vector
    R_inv = R.T        
    t_inv = -R_inv @ t
    # Construct the world-to-camera matrix
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_inv
    T_wc[:3, 3] = t_inv
    return T_wc

def read_mat_txt(path):
    with open(path, 'r') as file:
        data = file.read().split()
        data = [float(f) for f in data]
        mat = np.array(data).reshape((4,4))
    return mat

def read_pairs_scene_info(model_path, views, feature_name, img_name):
    view = views[0]
    test_path = Path(model_path).parent.parent.parent/'test_pairs'
    color_path = test_path/'images'
    pose_path = test_path/'pose'
    feature_path = test_path/f"features/{feature_name.split('/')[-1]}"
    new_views = []
    names = os.listdir(color_path)
    names = [name.split('.')[0] for name in names]
    names = sorted(names, key=lambda name:int(name))
    x=[views[i].image_name for i in range(len(views))]
    for i, name in enumerate(names):
        view = views[i]
        c_path = color_path/f"{name}.jpg"
        p_path = pose_path/f"{name}.txt"
        img = Image.open(c_path)

        if img_name=="images":
            resized_image_rgb = PILtoTorch(img, resolution=[1296, 968])
        elif img_name=="images_s2":
            resized_image_rgb = PILtoTorch(img, resolution=[int(1296/2), int(968/2)])
        extr = c2w_to_w2c(read_mat_txt(p_path))
        R = np.transpose(extr[:3, :3])
        T = np.array(extr[:3, 3])
        view.R = R
        view.T = T
        view.original_image = resized_image_rgb.clamp(0.0, 1.0).to(view.data_device)
        view.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        view.extrinsic_matrix = getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)
        view.image_name = name
        new_views.append(view)
    return new_views

def render_pairs(model_path, feature_name, img_name, name, iteration, views, gaussians, pipe_param, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "image_renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "image_gt")
    
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_renders")
    saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_tensors")
    score_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "score_renders")
    saved_score_path = os.path.join(model_path, name, "ours_{}".format(iteration), "score_tensors")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(feature_map_path, exist_ok=True)
    makedirs(saved_feature_path, exist_ok=True)
    makedirs(score_map_path, exist_ok=True)
    makedirs(saved_score_path, exist_ok=True)
    new_views = read_pairs_scene_info(model_path, views, feature_name, img_name)

    for _, view in enumerate(tqdm(new_views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipe_param, background)
        gt = view.original_image[0:3, :, :]
        gt_feature = view.semantic_feature
        gt_score = view.score_feature
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{}'.format(view.image_name) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))
        ############## visualize feature map
        feature_map = render_pkg["feature_map"]
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature.shape[1], gt_feature.shape[2]), 
                                    mode='bilinear', align_corners=True).squeeze(0) ###

        feature_map_vis = feature_visualize_saving(feature_map)
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(
                                            os.path.join(feature_map_path, '{}'.format(view.image_name) + "_feature_vis.png"))
        

        feature_map = feature_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{}'.format(view.image_name) + "_fmap.pt"))
        
        ############# score map
        score_map = render_pkg['score_map']
        # gt_score_map = view.score_feature.cuda()
        score_map = F.interpolate(score_map.unsqueeze(0), size=(gt_score.shape[1], gt_score.shape[2]), 
                                  mode='bilinear', align_corners=True).squeeze(0) ###
        score_map_vis = one_channel_vis(score_map)
        score_map_vis.save(os.path.join(score_map_path, '{}'.format(view.image_name) + "_score_vis.png"))

        score_map = score_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(score_map).half(), os.path.join(saved_score_path, '{}'.format(view.image_name) + "_smap.pt"))
        #############



# render all
def render_sets(model_param : ModelParams, iteration : int, pipe_param : PipelineParams, args):
    with torch.no_grad():
        gaussians = GaussianModel(model_param.sh_degree)
        if args.view_num==-1:
            args.view_num = None
        scene = Scene(model_param, gaussians, load_iteration=iteration, shuffle=False, view_num=args.view_num, 
                      load_feature=True)
        conf = {
            "sparse_outputs": True,
            "dense_outputs": True,
            "max_num_keypoints": args.sp_kpt,
            "detection_threshold": float(args.sp_th),
        }
        render_encoder = SuperPoint(conf).cuda().eval()
        bg_color = [1,1,1] if model_param.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if args.render_train:
            render_set(model_param.model_path, "rendering/trains", scene.loaded_iter, 
                       scene.getTrainCameras(), 
                       gaussians, pipe_param, background, args, scene.mlp, render_encoder)
        if args.render_test:
            render_set(model_param.model_path, "rendering/test", scene.loaded_iter, 
                       scene.getTestCameras(), 
                       gaussians, pipe_param, background, args, scene.mlp, render_encoder)
        if args.novel_view:
            render_novel_views(model_param.model_path, "rendering/novel_views", scene.loaded_iter, scene.getTrainCameras(), 
                               gaussians, pipe_param, background, model_param.speedup, 
                                args.multi_interpolate)
        if args.pairs:
            render_pairs(model_param.model_path, model_param.foundation_model, "", "rendering/pairs", scene.loaded_iter, 
                         scene.getTrainCameras(), gaussians, pipe_param, background)



# ( bash z_scripts/train.sh )
if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--detector_path", default=None, type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--view_num", default=-1, type=int)
    parser.add_argument("--lg_th", default=0.01, type=float)
    parser.add_argument("--hist", default=0.95, type=float)
    parser.add_argument("--sp_kpt", default=512, type=int)
    parser.add_argument("--sp_th", default=0.005, type=float)
    parser.add_argument("--match_precision_th", default=5e-6, type=float)
    # render cams
    parser.add_argument("--render_train", action="store_true")
    parser.add_argument("--render_test", action="store_true")
    parser.add_argument("--render_kpt_desc", action="store_true")
    parser.add_argument("--render_match", action="store_true")
    # render new types
    parser.add_argument("--novel_view", action="store_true") ###
    parser.add_argument("--multi_interpolate", action="store_true") ###
    parser.add_argument("--pairs", action="store_true")
    # RNG
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    print(args.source_path)
    safe_state(args.quiet)
    render_sets(Model_param.extract(args), args.iteration, Pipe_param.extract(args), args)
