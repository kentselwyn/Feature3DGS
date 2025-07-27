import os
import cv2
import time
import copy
import torch
import random
import dsacstar
import numpy as np
from pathlib import Path
from data.ace.dataset import CamLocDataset
from data.ace.ace_network import Regressor
from argparse import ArgumentParser
import utils.loc.loc_utils as loc_utils
from torch.cuda.amp import autocast
from scene import Scene
from scene.gaussian.gaussian_model import GaussianModel
from torch.utils.data import DataLoader
from utils.graphics_utils import fov2focal
from utils.loc.depth import project_2d_to_3d
from gaussian_renderer import render_gsplat
from matchers.lightglue import LightGlue
from encoders.superpoint.superpoint import SuperPoint
from utils.loc.pycolmap_utils import opencv_to_pycolmap_pnp
from arguments import ModelParams, PipelineParams, get_combined_args
from mlp.mlp import get_mlp_new
from datetime import datetime
from utils.match.match_img import save_matchimg_th
from utils.match.metrics_match import compute_metrics
from PIL import Image


def log_all_err(log_path, index, name,
                rotError, traError, rotError_two, traError_two,
                elapsed_time, 
                rotError_third=None, traError_third=None,
                rotError_forth=None, traError_forth=None, ):
    with open(log_path, 'a') as f:
        f.write(f"{'-'*40}\n")
        f.write(f"{index}, {name}, {elapsed_time*1000:.1f}ms\n")
        if rotError_third is None:
            f.write(f"{rotError:.2f},  {rotError_two:.2f}\n")
            f.write(f"{traError:.2f},  {traError_two:.2f}\n")
        else:
            if rotError_forth is None:
                f.write(f"{rotError:.2f},  {rotError_two:.2f},  {rotError_third:.2f}\n")
                f.write(f"{traError:.2f},  {traError_two:.2f},  {traError_third:.2f}\n")
            else:
                f.write(f"{rotError:.2f},  {rotError_two:.2f},  {rotError_third:.2f},  {rotError_forth:.2f}\n")
                f.write(f"{traError:.2f},  {traError_two:.2f},  {traError_third:.2f},  {traError_forth:.2f}\n")
        f.write(f"{'-'*40}\n\n")


def refine_img(img_gt0, img_gt_render0, img_render1):
    # img_gt_np0 = (img_gt0.permute(1, 2, 0)* 255).clamp(0, 255).byte().detach().cpu().numpy()
    # img_gt_render_np0 = (img_gt_render0.permute(1, 2, 0)* 255).clamp(0, 255).byte().detach().cpu().numpy()
    # img_render_np1 = (img_render1.permute(1, 2, 0)* 255).clamp(0, 255).byte().detach().cpu().numpy()
    # Image.fromarray(img_gt_np0).save("not_used/img_gt_np_0.png")
    # Image.fromarray(img_gt_render_np0).save("not_used/img_gtrender_np_0.png")
    # Image.fromarray(img_render_np1).save("not_used/img_render_np_1.png")
    fused_01 = img_gt0*0.5 + img_render1*0.5
    fused_01 = fused_01.unsqueeze(0)
    # fused_np01 = (fused_01.permute(1, 2, 0)* 255).clamp(0, 255).byte().detach().cpu().numpy()
    # Image.fromarray(fused_np01).save("not_used/fused_np01.png")
    return fused_01


random.seed(100)
def localize_set(args, 
                 error_foler_path, all_err_log_path,
                 views,            gaussians, pipe_param, background, 
                 encoder,          matcher, 
                 ace_network,      ace_test_loader,
                 match_folder_path=None,
                 render_folder_path=None):
    rErrs = []
    tErrs = []
    prior_rErr = []
    prior_tErr = []
    third_rErr = []
    third_tErr = []
    fourth_rErr = []
    fourth_tErr = []
    total_elapsed_time = 0
    mlp = get_mlp_new(dim=args.mlp_dim, name=args.method).cuda().eval()
    with torch.no_grad():
        for index, (image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames) in enumerate(ace_test_loader):
            start = time.time()
            image_B1HW = image_B1HW.to(torch.device("cuda"), non_blocking=True)
            with autocast(enabled=True):
                scene_coordinates_B3HW = ace_network(image_B1HW)
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()
            ########################################################
            for _, (scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path) in \
                            enumerate(zip(scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames)):
                ########################################################
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])
                out_pose = torch.zeros((4, 4))
                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0), out_pose, 64, 10,
                    focal_length, ppX, ppY, 100, 100, ace_network.OUTPUT_SUBSAMPLE,)
                rotError, transError = loc_utils.calculate_pose_errors_ace(
                    gt_pose_44, out_pose)
                ########################################################
                view = views[index]
                print(f"{index}, {view.image_name}")
                
                K = np.eye(3)
                focal_length = fov2focal(view.FoVx, view.image_width)
                K[0, 0] = K[1, 1] = focal_length
                K[0, 2] = view.image_width / 2
                K[1, 2] = view.image_height / 2
                gt_R = view.R
                gt_t = view.T
                gt_extrinsic_matrix = view.extrinsic_matrix
                ########################################################
                if args.match_type==3 or args.match_type==4 or (args.save_render_img is not None):
                    render_pkg0 = render_gsplat(view, gaussians, background, rgb_only=False)
                ########################################################
                out_R = out_pose[0:3, 0:3].numpy()
                out_t = out_pose[0:3, 3].numpy()
                R_inv = out_R.T
                t_inv = -R_inv @ out_t
                w2c = torch.eye(4, 4, device='cuda')
                w2c[:3, :3] = torch.from_numpy(R_inv).float()
                w2c[:3, 3] = torch.from_numpy(t_inv).float()
                view.update_RT(out_R, t_inv)
                render_pkg1 = render_gsplat(view, gaussians, background, rgb_only=False)
                ########################################################
                gt_img0 = view.original_image[0:3, :, :].cuda().unsqueeze(0)
                # 0 ours
                if args.match_type==0:
                    result = loc_utils.img_match_ours(args, gt_img0, 
                                                      render_pkg1["score_map"], render_pkg1["feature_map"], 
                                                      encoder, matcher, mlp)
                # 1 rival
                elif args.match_type==1 or args.match_type==6 or args.match_type==7 or args.match_type==8:
                    result = loc_utils.img_match_rival(gt_img0, render_pkg1["render"].unsqueeze(0), 
                                                       encoder, matcher)
                # 2 Renderkpt featSP
                elif args.match_type==2:
                    result = loc_utils.img_match_kptSPfeat(args, gt_img0, render_pkg1["render"].unsqueeze(0), 
                                                           render_pkg1["score_map"],
                                                           encoder, matcher)
                # 3 nothappen ours
                elif args.match_type==3:
                    result = loc_utils.img_match_RenderRender(args, render_pkg0["score_map"], render_pkg0["feature_map"],
                                                              render_pkg1["score_map"], render_pkg1["feature_map"],
                                                              matcher, mlp, )
                # 4 nothappen rival
                elif args.match_type==4:
                    result = loc_utils.img_match_rival(render_pkg0["render"].unsqueeze(0), 
                                                       render_pkg1["render"].unsqueeze(0), 
                                                       encoder, matcher)
                # 5 rival image blur
                elif args.match_type==5:
                    new_gtimg0 = refine_img(gt_img0.squeeze(0), render_pkg0["render"], render_pkg1["render"])
                    result = loc_utils.img_match_rival(new_gtimg0, render_pkg1["render"].unsqueeze(0), 
                                                       encoder, matcher)
                ########################################################
                if result is None:
                    prior_rErr.append(rotError)
                    prior_tErr.append(transError)
                    rErrs.append(rotError)
                    tErrs.append(transError)
                    print(f"Rotation Error: {rotError} deg")
                    print(f"Translation Error: {transError} cm")
                    total_elapsed_time += time.time()-start
                    continue
                if not len(result['mkpt1'].cpu())>args.stop_kpt_num:
                    prior_rErr.append(rotError)
                    prior_tErr.append(transError)
                    rErrs.append(rotError)
                    tErrs.append(transError)
                    print(f"Rotation Error: {rotError} deg")
                    print(f"Translation Error: {transError} cm")
                    total_elapsed_time += time.time()-start
                    continue
                ########################################################
                world_points = project_2d_to_3d(result['mkpt1'].cpu(), render_pkg1["depth"].cpu(), 
                                            torch.tensor(K, dtype=torch.float32).cpu(), w2c.cpu())\
                                            .cpu().numpy().astype(np.float64)
                match0 = result['mkpt0'].cpu().numpy().astype(np.float64)
                if args.pnp == "iters":
                    _, R_final, t_final, _ = cv2.solvePnPRansac(world_points, match0, K, distCoeffs=None, 
                                                                flags=cv2.SOLVEPNP_ITERATIVE, 
                                                                iterationsCount=args.ransac_iters)
                    R_final, _ = cv2.Rodrigues(R_final)
                elif args.pnp == "epnp":
                    _, R_final, t_final, _ = cv2.solvePnPRansac(world_points, match0, K, distCoeffs=None, 
                                                                flags=cv2.SOLVEPNP_EPNP)
                    R_final, _ = cv2.Rodrigues(R_final)
                elif args.pnp == "pycolmap":
                    R_final, t_final = opencv_to_pycolmap_pnp(world_points, match0, K, 
                                                        view.image_width, view.image_height)
                ########################################################
                rotError_two, transError_two = loc_utils.calculate_pose_errors(gt_R, gt_t, R_final.T, t_final)
                rotError_final, transError_final = rotError_two, transError_two
                ########################################################
                if args.match_type==6 or args.match_type==7 or args.match_type==8:
                    update_R = R_final.T
                    update_t = t_final.squeeze(-1)
                    view.update_RT(update_R, update_t)
                    render_pkg1 = render_gsplat(view, gaussians, background, rgb_only=False)
                    # gt_img0, render_pkg1["render"], render_pkg2["render"]
                    if args.match_type==6:
                        match_012_final = loc_utils.img_match_circular(
                            result, gt_img0, render_pkg1["render"].unsqueeze(0), render_pkg1["render"].unsqueeze(0), 
                            encoder, matcher)
                    if args.match_type==7 or args.match_type==8:
                        match_012_final = loc_utils.img_match_rival(gt_img0, render_pkg1["render"].unsqueeze(0), 
                                                       encoder, matcher)
                    result = match_012_final
                    ###################
                    w2c_2 = torch.eye(4, 4, device='cuda')
                    w2c_2[:3, :3] = torch.from_numpy(update_R.T).float()
                    w2c_2[:3, 3] = torch.from_numpy(update_t).float()
                    ###################
                    world_points = project_2d_to_3d(result['mkpt1'].cpu(), render_pkg1["depth"].cpu(), 
                                            torch.tensor(K, dtype=torch.float32).cpu(), w2c_2.cpu()
                                            ).cpu().numpy().astype(np.float64)
                    match0 = result['mkpt0'].cpu().numpy().astype(np.float64)
                    R_third, t_third = opencv_to_pycolmap_pnp(world_points, match0, K, 
                                                        view.image_width, view.image_height)
                    rotError_third, transError_third = loc_utils.calculate_pose_errors(gt_R, gt_t, R_third.T, t_third)
                    rotError_final, transError_final = rotError_third, transError_third
                    ########################################################
                    if args.match_type==8:
                        update_R = R_third.T
                        update_t = t_third.squeeze(-1)
                        view.update_RT(update_R, update_t)
                        render_pkg1 = render_gsplat(view, gaussians, background, rgb_only=False)
                        match_012_final = loc_utils.img_match_rival(gt_img0, render_pkg1["render"].unsqueeze(0), 
                                                       encoder, matcher)
                        result = match_012_final
                        ###################
                        w2c_2 = torch.eye(4, 4, device='cuda')
                        w2c_2[:3, :3] = torch.from_numpy(update_R.T).float()
                        w2c_2[:3, 3] = torch.from_numpy(update_t).float()
                        world_points = project_2d_to_3d(result['mkpt1'].cpu(), render_pkg1["depth"].cpu(), 
                                            torch.tensor(K, dtype=torch.float32).cpu(), w2c_2.cpu()
                                            ).cpu().numpy().astype(np.float64)
                        match0 = result['mkpt0'].cpu().numpy().astype(np.float64)
                        R_fourth, t_fourth = opencv_to_pycolmap_pnp(world_points, match0, K, 
                                                        view.image_width, view.image_height)
                        rotError_four, transError_four = loc_utils.calculate_pose_errors(gt_R, gt_t, R_fourth.T, t_fourth)
                        rotError_final, transError_final = rotError_four, transError_four
                ########################################################
                if match_folder_path is not None:
                    if args.match_type==3 or args.match_type==4:
                        result['img0'] = render_pkg0["render"].permute(1, 2, 0)
                    elif args.match_type==5:
                        result['img0'] = new_gtimg0.squeeze(0).permute(1, 2, 0)
                    else:
                        result['img0'] = gt_img0.squeeze(0).permute(1, 2, 0)
                    ################
                    result['img1'] = render_pkg1["render"].permute(1, 2, 0)
                    ########################################################
                    T0 = gt_extrinsic_matrix
                    T1 = view.extrinsic_matrix
                    T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)
                    T_1to0 = T_0to1.inverse()
                    K = torch.tensor((views[0].intrinsic_matrix).astype(np.float32))
                    pose_data = {"K0": K, "K1": K,
                        "T_0to1": T_0to1.float(),
                        "T_1to0": T_1to0.float(), "identifiers": [f"{view.image_name}"],}
                    ########################################################
                    result.update(pose_data)
                    compute_metrics(result)
                    save_matchimg_th(result, 
                        f'{match_folder_path}/{index}_{view.image_name}__' + 
                        f'(T:{transError:.2f}_R:{rotError:.2f})__(T:{transError_final:.2f}_R:{rotError_final:.2f}).png',
                        threshold=5e-5)
                ########################################################
                if render_folder_path is not None:
                    ################ gt, rendergt 
                    gt_img = gt_img0.squeeze(0).permute(1, 2, 0)
                    render_img_gt = render_pkg0["render"].permute(1, 2, 0)
                    ################
                    render_img1 = render_pkg1["render"].permute(1, 2, 0)
                    ################
                    update_R = R_final.T
                    update_t = t_final.squeeze(-1)
                    new_view = copy.deepcopy(view)
                    new_view.update_RT(update_R, update_t)
                    render_pkg_final = render_gsplat(new_view, gaussians, background, rgb_only=False)
                    render_img_final = render_pkg_final["render"].permute(1, 2, 0)
                    ################
                    top = torch.cat((gt_img, render_img_gt), dim=1)
                    bottom = torch.cat((render_img1, render_img_final), dim=1)
                    combined = torch.cat((top, bottom), dim=0)
                    ################
                    combined_np = (combined * 255).clamp(0, 255).byte().detach().cpu().numpy()
                    Image.fromarray(combined_np).save(f"{render_folder_path}/{index}_{view.image_name}__"+
                                                      f"(T:{transError:.2f}_R:{rotError:.2f})__(T:{transError_final:.2f}_R:{rotError_final:.2f}).png" )
                ########################################################
                
                print(f"{index}, {view.image_name}")
                print(f"Rotation Error: {rotError} deg")
                print(f"Translation Error: {transError} cm")
                print(f"Second Rotation Error: {rotError_two} deg")
                print(f"Second Translation Error: {transError_two} cm")
                if args.match_type==6 or args.match_type==7 or args.match_type==8:
                    print(f"Third Rotation Error: {rotError_third} deg")
                    print(f"Third Translation Error: {transError_third} cm")
                    if args.match_type==8:
                        print(f"fourth Rotation Error: {rotError_four} deg")
                        print(f"fourth Translation Error: {transError_four} cm")
                elapsed_time = time.time()-start
                total_elapsed_time += elapsed_time
                print(f"elapsed time: {elapsed_time}")
                print()
                if args.match_type==6 or args.match_type==7:
                    log_all_err(all_err_log_path, index, view.image_name, 
                            rotError, transError, rotError_two, transError_two,
                            elapsed_time, 
                            rotError_third=rotError_third, traError_third=transError_third)
                elif args.match_type==8:
                    log_all_err(all_err_log_path, index, view.image_name, 
                            rotError, transError, rotError_two, transError_two,
                            elapsed_time, 
                            rotError_third=rotError_third, traError_third=transError_third,
                            rotError_forth=rotError_four, traError_forth=transError_four)
                else:
                    log_all_err(all_err_log_path, index, view.image_name, 
                            rotError, transError, rotError_two, transError_two,
                            elapsed_time)
                prior_rErr.append(rotError)
                prior_tErr.append(transError)
                rErrs.append(rotError_two)
                tErrs.append(transError_two)
                if args.match_type==6 or args.match_type==7 or args.match_type==8:
                    third_rErr.append(rotError_third)
                    third_tErr.append(transError_third)
                    if args.match_type==8:
                        fourth_rErr.append(rotError_four)
                        fourth_tErr.append(transError_four)
    mean_elapsed_time = total_elapsed_time / len(rErrs)
    print('rot len: ',len(prior_rErr))
    print('final rot len: ', len(rErrs))
    print('mean elapsed time: ', mean_elapsed_time)
    print()
    error_text = "prior_final"
    loc_utils.log_errors(error_foler_path, prior_rErr, prior_tErr, list_text="prior", error_text=error_text)
    loc_utils.log_errors(error_foler_path, rErrs,      tErrs,      list_text="warp",   error_text=error_text, 
                         elapsed_time=mean_elapsed_time)
    if args.match_type==6 or args.match_type==7 or args.match_type==8:
        loc_utils.log_errors(error_foler_path, third_rErr, third_tErr,  list_text="third",   error_text=error_text, 
                         elapsed_time=mean_elapsed_time)
        if args.match_type==8:
            loc_utils.log_errors(error_foler_path, fourth_rErr, fourth_tErr,  list_text="fourth",   error_text=error_text, 
                                elapsed_time=mean_elapsed_time)

def localize(args, model_param:ModelParams, pipe_param:PipelineParams):
    gaussians = GaussianModel(model_param.sh_degree)
    scene = Scene(model_param, 
                  gaussians, 
                  load_iteration=args.iteration, 
                  shuffle=False, 
                  load_feature=False, load_train_cams=False)
    ########################################################
    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": args.max_num_kpt,
        "detection_threshold": args.sp_th,
    }
    encoder = SuperPoint(conf).cuda().eval()
    matcher = LightGlue({"filter_threshold": args.lg_th ,}).cuda().eval()
    ########################################################
    encoder_state_dict = torch.load(args.ace_encoder_path, map_location="cpu")
    head_state_dict = torch.load(args.ace_ckpt, map_location="cpu")
    ace_network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict).cuda().eval()
    testset = CamLocDataset(
        Path(args.source_path) / "test",
        mode=0,
        image_height=480,
    )
    ace_test_loader = DataLoader(testset, shuffle=False, num_workers=0)
    ########################################################
    match_type_dict = {
        0: "ours",
        1: "rival",
        2: "Renderkpt_featSP",
        3: "nothappen_ours",
        4: "nothappen_rival",
        5: "rival_image_blur",
        6: "circular_match",
        7: "two_times_matches",
        8: "three_times_matches",
    }
    loc_path = Path(model_param.model_path)/"localization"/f"{args.match_type}_{match_type_dict[args.match_type]}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ##########################
    if args.save_match:
        match_folder_path = f'{loc_path}/{args.test_name}/match_imgs_{timestamp}'
        os.makedirs(match_folder_path, exist_ok=True)
    else:
        match_folder_path = None
    ##########################
    if args.save_render_img:
        render_folder_path = f'{loc_path}/{args.test_name}/render_imgs_{timestamp}'
        os.makedirs(render_folder_path, exist_ok=True)
    else:
        render_folder_path = None
    ##########################
    error_foler_path = f'{loc_path}/{args.test_name}/error_logs'
    all_err_log_path = f"{str(Path(error_foler_path).parent)}/log_{timestamp}.txt"
    os.makedirs(error_foler_path, exist_ok=True)
    if os.path.exists(all_err_log_path):
        os.remove(all_err_log_path)
    ########################################################
    bg_color = [1,1,1] if model_param.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    localize_set(args, 
                 error_foler_path, all_err_log_path,
                 scene.getTestCameras(), 
                 gaussians,   pipe_param,  background,
                 encoder,     matcher, 
                 ace_network, ace_test_loader,
                 match_folder_path=match_folder_path,
                 render_folder_path=render_folder_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--test_name", required=True, type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--method",   required=True, type=str)
    parser.add_argument("--ace_ckpt", required=True, type=str)
    parser.add_argument("--save_match", action='store_true')
    parser.add_argument("--save_render_img", action='store_true')
    # used for match
    parser.add_argument("--match_type", default=0, type=int)
    parser.add_argument("--sp_th", default=0.0, type=float)
    parser.add_argument("--lg_th", default=0.01, type=float)
    parser.add_argument("--kpt_hist", default=0.95, type=float)
    parser.add_argument("--kpt_th", default=0.01, type=float)
    parser.add_argument("--kernel_size", default=15, type=int)
    parser.add_argument("--max_num_kpt", default=1024, type=int)
    # ransac config
    parser.add_argument("--ransac_iters", default=20000, type=int)
    parser.add_argument("--stop_kpt_num", default=30, type=int)
    parser.add_argument("--pnp", default="pycolmap", type=str)
    parser.add_argument("--ace_encoder_path", 
        default="/home/koki/code/cc/feature_3dgs_2/data/ace/ace_encoder_pretrained.pt", type=str)
    args = get_combined_args(parser)
    localize(args, Model_param.extract(args), Pipe_param.extract(args))
