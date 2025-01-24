import time
import torch
import numpy as np
from utils.loc_utils import *
from argparse import ArgumentParser
from gaussian_renderer.__init__median import render
from scene import Scene, GaussianModel
from find_depth import project_2d_to_3d

from utils.graphics_utils import fov2focal
from utils.vis_scoremap import one_channel_vis
from encoders.superpoint.mlp import get_mlp_dataset
from encoders.superpoint.lightglue import LightGlue
from encoders.superpoint.superpoint import SuperPoint
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.match_img import extract_kpt, sample_descriptors_fix_sampling, save_matchimg


def choose_th(score, histogram_th):
    score_flat = score.flatten()
    percentile_value = torch.quantile(score_flat, float(histogram_th))
    return percentile_value.item()


def find_filtered_gaussian(gaussian_pcd, gaussian_feat_h, scores):
    th = choose_th(scores, 0.98)
    mask_score = scores>th
    sum_score = mask_score.sum()
    mask_score = mask_score.squeeze(-1)
    filtered_points = gaussian_pcd[mask_score]
    filtered_feature = gaussian_feat_h[mask_score]
    return filtered_points, filtered_feature




def localize_set(model_path, name, views, gaussians, pipe_param, 
                 background, args, encoder, matcher):
    rErrs = []
    tErrs = []

    prior_rErr = []
    prior_tErr = []

    scene_name = model_path.split('/')[-3]
    print(scene_name)
    # breakpoint()
    mlp = get_mlp_dataset(dataset=scene_name).cuda().eval()

    for index, _ in enumerate(views):
        # index = 151
        view = views[index]
        start = time.time()
        gaussian_pcd = gaussians.get_xyz
        gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
        scores = gaussians.get_score_feature.squeeze(-1)

        gt_im = view.original_image[0:3, :, :].cuda().unsqueeze(0)
        data = {}
        data["image"] = gt_im
        pred = encoder(data)
        

        gt_keypoints = pred["keypoints"].squeeze(0)
        desc = pred["descriptors"].squeeze(0)
        
        
        gt_feature = mlp.decode(mlp(desc))


        gaussian_feat_h = mlp.decode(gaussian_feat)
        

        K = np.eye(3)
        focal_length = fov2focal(view.FoVx, view.image_width)
        K[0, 0] = K[1, 1] = focal_length
        K[0, 2] = view.image_width / 2
        K[1, 2] = view.image_height / 2

        start = time.time()
        with torch.no_grad():
            gaussian_pcd, gaussian_feat_h = find_filtered_gaussian(gaussian_pcd.detach(), 
                                                                   gaussian_feat_h.detach(), 
                                                                   scores.detach())
            matched_3d, matched_2d = find_2d3d_correspondences(gt_keypoints.detach(), 
                                                            gt_feature.detach(), 
                                                            gaussian_pcd.detach(), 
                                                            gaussian_feat_h.detach())
        gt_R = view.R
        gt_t = view.T
        # print(f"Match speed: {time.time() - start}")
        _, R, t, _ = cv2.solvePnPRansac(matched_3d, matched_2d, 
                                        K, 
                                        distCoeffs=None, 
                                        flags=cv2.SOLVEPNP_ITERATIVE, 
                                        iterationsCount=args.ransac_iters
                                        )
        # breakpoint()
        R, _ = cv2.Rodrigues(R)
        # Calculate the rotation and translation errors using existing function
        rotError, transError = calculate_pose_errors(gt_R, gt_t, R.T, t)

        # Print the errors
        print(f"{index}, {view.image_name}")
        print(f"Rotation Error: {rotError} deg")
        print(f"Translation Error: {transError} cm")
        prior_rErr.append(rotError)
        prior_tErr.append(transError)


        w2c = torch.eye(4, 4, device='cuda')
        w2c[:3, :3] = torch.from_numpy(R).float()
        w2c[:3, 3] = torch.from_numpy(t[:, 0]).float()
        
        # Update the view's pose
        view.update_RT(R.T, t[:,0])
        
        # Render from the current estimated pose
        with torch.no_grad():
            render_pkg = render(view, gaussians, pipe_param, background)
        
        db_render = render_pkg["render"]
        db_score = render_pkg["score_map"]
        db_feature = render_pkg["feature_map"]
        db_depth = render_pkg["depth"]

        query_render = gt_im

        # quat_opt = rotmat2qvec_tensor(w2c[:3, :3].clone()).view([4]).to(w2c.device)
        # t_opt = w2c[:3, 3].clone()
        result = match_img(query_render, db_score, db_feature, encoder, matcher, mlp)
        if result is None:
            continue
        if not len(result['mkpt1'].cpu())>4:
            continue

        result['img0'] = gt_im.squeeze(0).permute(1, 2, 0)
        result['img1'] = db_render.squeeze(0).permute(1, 2, 0)
        # save_matchimg(result, 'match.png')
        # K = torch.tensor(K, dtype=torch.float32)
        db_world = project_2d_to_3d(result['mkpt1'].cpu(), db_depth.cpu(), torch.tensor(K, dtype=torch.float32).cpu(), w2c.cpu()).cpu().numpy().astype(np.float64)
        q_matched = result['mkpt0'].cpu().numpy().astype(np.float64)
        _, R_final, t_final, _ = cv2.solvePnPRansac(db_world, q_matched, K, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=args.ransac_iters)
        R_final, _ = cv2.Rodrigues(R_final)
        rotError_final, transError_final = calculate_pose_errors(gt_R, gt_t, R_final.T, t_final)

        # Print the errors
        print(f"Final Rotation Error: {rotError_final} deg")
        print(f"Final Translation Error: {transError_final} cm")
        print(f"elapsed time: {time.time()-start}")
        
        prior_rErr.append(rotError)
        prior_tErr.append(transError)
        rErrs.append(rotError_final)
        tErrs.append(transError_final)

        print()
        # if index==6:
        #     break
        # break

    log_errors(model_path, name, prior_rErr, prior_tErr, f"prior")
    log_errors(model_path, name, rErrs, tErrs, "warp")




def match_img(render_q, score_db, feature_db, encoder, matcher, mlp):
    # score map vis
    # score_map_vis = one_channel_vis(score_db)
    # score_map_vis.save("score_vis.png")

    tmp = {}
    tmp["image"] = render_q
    tmp_pred = encoder(tmp)
    desc = tmp_pred["descriptors"]
    gt_feature = mlp.decode(mlp(desc))

    # query
    kpt_q = tmp_pred["keypoints"]
    feat_q = gt_feature
    
    # db
    kpt_db = extract_kpt(score_db, threshold=0.3)
    if kpt_db.shape[0] == 0:
        return None
    kpt_db = kpt_db.clone().detach()[:, [1, 0]].to(score_db)
    _, h, w = score_db.shape
    scale = torch.tensor([w, h]).to(score_db)
    feat_db = sample_descriptors_fix_sampling(kpt_db, feature_db, scale)
    feat_db = mlp.decode(feat_db)
    kpt_db = kpt_db.unsqueeze(0)
    
    # match
    data = {}
    data["keypoints0"] = kpt_q
    data["keypoints1"] = kpt_db
    data["descriptors0"] = feat_q
    data["descriptors1"] = feat_db
    data["image_size"] = score_db.shape[1:]
    pred = matcher(data)

    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = data["keypoints0"][0][valid].cpu(), data["keypoints1"][0][m0[0][valid]].cpu()

    result = {}
    result['mkpt0'] = m0
    result['mkpt1'] = m1
    result['kpt0'] = kpt_q[0].cpu()
    result['kpt1'] = kpt_db[0].cpu()

    return result




def localize(model_param:ModelParams, pipe_param:PipelineParams, args):
    gaussians = GaussianModel(model_param.sh_degree)
    scene = Scene(model_param, gaussians, load_iteration=args.iteration, shuffle=False, load_feature=False)
    bg_color = [1,1,1] if model_param.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": 1024,
        "detection_threshold": 0.005,
    }
    encoder = SuperPoint(conf).cuda().eval()
    matcher = LightGlue({"filter_threshold": 0.01 ,}).cuda().eval()

    localize_set(model_param.model_path, "test", scene.getTestCameras(), 
                 gaussians, pipe_param, background, args, encoder, matcher)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ransac_iters", default=20000, type=int)
    parser.add_argument("--warp_lr", default=0.0005, type=float)
    parser.add_argument("--warp_iters", default=251, type=int)
    args = get_combined_args(parser)

    localize(Model_param.extract(args), Pipe_param.extract(args), args)
