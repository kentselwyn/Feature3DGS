import os
import torch
import open3d as o3d
###################################################
from scene import Scene
from scene.gaussian.gaussian_model_score import GaussianModel as Gauss_score
from scene.gaussian.gaussian_model_feature import GaussianModel as GaussianModelFeature
from gaussian_renderer import render_gsplat
###################################################
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.scoremap_vis import one_channel_vis
from tqdm import tqdm
import torch.nn.functional as F
from render import feature_visualize_saving
from PIL import Image
import numpy as np
from utils.match.match_img import find_small_circle_centers, sample_descriptors_fix_sampling, \
                                    save_matchimg, match_data
from mlp.mlp import get_mlp_dataset
from copy import deepcopy
from matchers.lightglue import LightGlue
from encoders.superpoint.superpoint import SuperPoint


def get_new_gaussians(filter_th):
    gaussians = Gauss_score(3)
    model_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs/outputs/twoPhase_all_weighted0.2_scale0.6"
    gaussians.load_ply(os.path.join(model_path,
                                    "point_cloud",
                                    "iteration_" + str(10000),
                                    "point_cloud.ply"))
    xyz = gaussians.get_xyz
    scores = gaussians.get_score_feature
    flat_scores = scores.view(-1) 
    threshold = torch.quantile(flat_scores, filter_th)
    mask = flat_scores.detach() > threshold
    filtered_xyz = xyz[mask]
    filtered_scores = scores[mask.view(-1, 1, 1)]
    points_np = filtered_xyz.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    gaussians._xyz = gaussians._xyz[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._rotation = gaussians._rotation[mask]
    gaussians._opacity = gaussians._opacity[mask]
    gaussians._score_feature = gaussians._score_feature[mask]
    return gaussians


def choose_th(feature, histogram_th):
    score_flat = feature.flatten()
    percentile_value = torch.quantile(score_flat, float(histogram_th))
    return percentile_value.item()


# ( bash zenith_scripts/twoPhase.sh )
if __name__=="__main__":
    # filter_th = 0
    parser = ArgumentParser(description="Testing script parameters")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_score", action="store_true")
    parser.add_argument("--render_desc", action="store_true")
    parser.add_argument("--render_match", action="store_true")
    parser.add_argument("--render_match_gt", action="store_true")
    parser.add_argument("--view_num", default=-1, type=int)
    args = get_combined_args(parser)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussians = GaussianModelFeature(3)
    scene = Scene(Model_param.extract(args), gaussians, shuffle=False, view_num=args.view_num, 
                      load_feature=True, load_iteration=args.iteration)
    
    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": 1024,
        "detection_threshold": 0.0,
    }
    encoder = SuperPoint(conf).to("cuda").eval()
    lg_th = 0.01
    my_hist = 0.9
    matcher = LightGlue({"filter_threshold": lg_th ,}).cuda().eval()
    mlp = get_mlp_dataset(16, dataset="pgt_7scenes_stairs").cuda().eval()

    views = scene.getTestCameras()

    out_path = os.path.join(args.model_path, "viz_preds")
    out_path_gt = os.path.join(args.model_path, "viz_gt")
    if args.render_score:
        out_path   +="_score"
        out_path_gt+="_score"
    if args.render_desc:
        out_path   +="_desc"
        out_path_gt+="_desc"
    if args.render_match or args.render_match_gt:
        out_path   +=f"_match_{args.view_num}_lg{lg_th}_myHist{my_hist}"
        out_path_gt+=f"_match_{args.view_num}_{conf['max_num_keypoints']}_{conf['detection_threshold']}_lg{lg_th}_myHist{my_hist}"
        out_path_gt_decode = out_path_gt + f"_decode"
        out_path_gt_sp = out_path_gt + f"_sp"
        out_path_render_sp = out_path_gt + f"_sp_render"
        os.makedirs(out_path_gt_decode, exist_ok=True)
        os.makedirs(out_path_gt_sp, exist_ok=True)
        os.makedirs(out_path_render_sp, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path_gt, exist_ok=True)

    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            render_pkg = render_gsplat(view, gaussians, background, rgb_only=False)

            if args.render_score:
                score_map = render_pkg['score_map']
                gt_score_map = view.score_feature.cuda()
                score_map_vis = one_channel_vis(score_map)
                gt_score_map_vis = one_channel_vis(gt_score_map)
                score_map_vis.save(f"{out_path}/{view.image_name}.png")
                gt_score_map_vis.save(f"{out_path_gt}/{view.image_name}.png")
            if args.render_desc:
                feature_map = render_pkg['feature_map']
                gt_feature_map = view.semantic_feature.cuda()
                feature_map = F.interpolate(feature_map.unsqueeze(0), 
                                            size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), 
                                            mode='bilinear', align_corners=True).squeeze(0)
                
                feature_map_vis = feature_visualize_saving(feature_map)
                Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(
                    f"{out_path}/{view.image_name}.png"))
                
                gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
                Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(
                    f"{out_path_gt}/{view.image_name}.png"))
        if args.render_match:
            pkg0 = render_gsplat(views[0], gaussians, background, rgb_only=False)
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                if idx==0:
                    continue
                pkg1 = render_gsplat(view, gaussians, background, rgb_only=False)
                s0, s1, f0, f1 = pkg0['score_map'], pkg1['score_map'], \
                                    pkg0['feature_map'], pkg1['feature_map']
                
                th0 = choose_th(s0, my_hist)
                th1 = choose_th(s1, my_hist)
                kpt0 = find_small_circle_centers(s0, threshold=th0, kernel_size=15).clone().detach()[:, [1, 0]]
                kpt1 = find_small_circle_centers(s1, threshold=th1, kernel_size=15).clone().detach()[:, [1, 0]]
                
                _, h, w = s0.shape
                scale = torch.tensor([w, h]).to(s0)
                feat0 = sample_descriptors_fix_sampling(kpt0, f0, scale)
                feat1 = sample_descriptors_fix_sampling(kpt1, f1, scale)
                
                feat0 = mlp.decode(feat0)
                feat1 = mlp.decode(feat1)
                ##############################################
                data = {}
                data["keypoints0"] = kpt0.unsqueeze(0)
                data["keypoints1"] = kpt1.unsqueeze(0)
                data["descriptors0"] = feat0
                data["descriptors1"] = feat1
                data["image_size"] = s0.shape[1:]

                result = match_data(data, matcher, 
                                    img0=pkg0['render'].permute(1, 2, 0),
                                    img1=pkg1['render'].permute(1, 2, 0))
                save_matchimg(result, 
                              f"{out_path}/{view.image_name}.png")
                ##############################################
                img_render0 = pkg0['render']
                img_render1 = pkg1['render']
                img_gt1 = view.original_image
                ##############################################
                tmp = {}
                tmp["image"] = img_gt1.unsqueeze(0).cuda()
                tmp_pred_gt1 = encoder(tmp)
                data = {}
                data["keypoints0"] = kpt0.unsqueeze(0)
                data["keypoints1"] = tmp_pred_gt1["keypoints"]
                data["descriptors0"] = feat0
                data["descriptors1"] = tmp_pred_gt1["descriptors"]
                data["image_size"] = s0.shape[1:]
                result = match_data(data, matcher, 
                                    img0=img_render0.permute(1, 2, 0),
                                    img1=img_gt1.permute(1, 2, 0))
                save_matchimg(result, 
                              f"{out_path_gt}/{view.image_name}.png")
                ##############################################
                data = {}
                data["keypoints0"] = kpt0.unsqueeze(0)
                data["keypoints1"] = tmp_pred_gt1["keypoints"]
                data["descriptors0"] = feat0
                data["descriptors1"] = mlp.decode(mlp(tmp_pred_gt1["descriptors"]))
                data["image_size"] = s0.shape[1:]
                result = match_data(data, matcher, 
                                    img0=img_render0.permute(1, 2, 0),
                                    img1=img_gt1.permute(1, 2, 0))
                save_matchimg(result, 
                              f"{out_path_gt_decode}/{view.image_name}.png")
                ##############################################
                tmp = {}
                tmp["image"] = img_render0.unsqueeze(0).cuda()
                tmp_pred0 = encoder(tmp)
                data = {}
                data["keypoints0"] = tmp_pred0["keypoints"]
                data["keypoints1"] = tmp_pred_gt1["keypoints"]
                data["descriptors0"] = tmp_pred0["descriptors"]
                data["descriptors1"] = tmp_pred_gt1["descriptors"]
                data["image_size"] = s0.shape[1:]
                result = match_data(data, matcher, 
                                    img0=img_render0.permute(1, 2, 0),
                                    img1=img_gt1.permute(1, 2, 0))
                save_matchimg(result, 
                              f"{out_path_gt_sp}/{view.image_name}.png")
                ##############################################
                tmp = {}
                tmp["image"] = img_render1.unsqueeze(0).cuda()
                tmp_pred1 = encoder(tmp)
                data = {}
                data["keypoints0"] = tmp_pred0["keypoints"]
                data["keypoints1"] = tmp_pred1["keypoints"]
                data["descriptors0"] = tmp_pred0["descriptors"]
                data["descriptors1"] = tmp_pred1["descriptors"]
                data["image_size"] = s0.shape[1:]
                result = match_data(data, matcher, 
                                    img0=img_render0.permute(1, 2, 0),
                                    img1=img_render1.permute(1, 2, 0))
                save_matchimg(result, 
                              f"{out_path_render_sp}/{view.image_name}.png")
                ##############################################
                pkg0 = deepcopy(pkg1)
