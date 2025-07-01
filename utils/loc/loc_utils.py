import os
import numpy as np
import cv2
import math
import time
import torch
import pickle
import open3d as o3d
from typing import Tuple
from matchers.LoFTR.utils.utils import rgb2loftrgray
import torch.nn.functional as F
from utils.match.match_img import sample_descriptors_fix_sampling, \
                                    find_small_circle_centers
from matchers.mast3r.mast3r.fast_nn import fast_reciprocal_NNs
from matchers.mast3r.dust3r.dust3r.inference import inference
from matchers.mast3r.dust3r.dust3r.utils.image import convert_tensor_to_dust3r_format
from matchers.mast3r.utils.functions import *
from utils.match.match_img import semi_img_match
from pathlib import Path


# log, calculate error
def calculate_pose_errors(R_gt, t_gt, R_est, t_est):
    rotError = np.matmul(R_est.T, R_gt)
    rotError = cv2.Rodrigues(rotError)[0]
    rotError = np.linalg.norm(rotError) * 180 / np.pi
    transError = np.linalg.norm(t_gt - t_est.squeeze(1)) * 100  # Convert to cm
    return rotError, transError


def calculate_pose_errors_ace(gt_pose_44, out_pose):
    t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))*100
    gt_R = gt_pose_44[0:3, 0:3].numpy()
    out_R = out_pose[0:3, 0:3].numpy()
    r_err = np.matmul(out_R, np.transpose(gt_R))
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi
    return r_err, t_err


def log_errors(log_dir, rotation_errors, translation_errors, 
                        list_text,       error_text,        elapsed_time=None):
    total_frames = len(rotation_errors)
    # Remove NaN values from rotation_errors and translation_errors
    rotation_errors = [err for err in rotation_errors if not np.isnan(err)]
    translation_errors = [err for err in translation_errors if not np.isnan(err)]

    min_length = min(len(rotation_errors), len(translation_errors))
    rotation_errors = rotation_errors[:min_length]
    translation_errors = translation_errors[:min_length]

    total_frames = len(rotation_errors)
    median_rErr = np.median(rotation_errors)
    median_tErr = np.median(translation_errors)
    
    pct10_5 = sum(r <= 5 and t <= 10 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100
    pct5 = sum(r <= 5 and t <= 5 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100
    pct2 = sum(r <= 2 and t <= 2 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100
    pct1 = sum(r <= 1 and t <= 1 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100
    print('Accuracy:')
    print(f'\t10cm/5deg: {pct10_5:.1f}%')
    print(f'\t5cm/5deg: {pct5:.1f}%')
    print(f'\t2cm/2deg: {pct2:.1f}%')
    print(f'\t1cm/1deg: {pct1:.1f}%')
    print(f'\tmedian_rErr: {median_rErr:.3f} deg')
    print(f'\tmedian_tErr: {median_tErr:.3f} cm')
    if elapsed_time is not None:
        print(f'Mean elapsed time: {elapsed_time:.6f} s\n')
    error_list = rotation_errors + translation_errors
    with open(os.path.join(log_dir, 
                           f'error_list_{list_text}.pickle'), 'wb') as f:
        pickle.dump(error_list, f)
    with open(os.path.join(str(Path(log_dir).parent), 
                           f'median_error_{error_text}_end.txt'), 'a') as f:
        f.write('Accuracy:\n')
        f.write(f'\t10cm/5deg: {pct10_5:.1f}%\n')
        f.write(f'\t5cm/5deg: {pct5:.1f}%\n')
        f.write(f'\t2cm/2deg: {pct2:.1f}%\n')
        f.write(f'\t1cm/1deg: {pct1:.1f}%\n')
        f.write(f'Median translation error: {median_tErr:.6f} cm\n')
        f.write(f'Median rotation error: {median_rErr:.6f} dg\n')
        if elapsed_time is not None:
            f.write(f'Mean elapsed time: {elapsed_time:.6f} s\n')




# find 2d, 3d correpondence
def find_2d3d_correspondences(keypoints, image_features, gaussian_pcd, gaussian_feat, chunk_size=10000):
    device = image_features.device
    f_N, feat_dim = image_features.shape
    P_N = gaussian_feat.shape[0]
    image_features = F.normalize(image_features, p=2, dim=1)
    gaussian_feat = F.normalize(gaussian_feat, p=2, dim=1)
    
    max_similarity = torch.full((f_N,), -float('inf'), device=device)
    max_indices = torch.zeros(f_N, dtype=torch.long, device=device)
    
    for part in range(0, P_N, chunk_size):
        chunk = gaussian_feat[part:part + chunk_size]
        similarity = torch.mm(image_features, chunk.t())
        chunk_max, chunk_indices = similarity.max(dim=1)
        update_mask = chunk_max > max_similarity
        max_similarity[update_mask] = chunk_max[update_mask]
        max_indices[update_mask] = chunk_indices[update_mask] + part
    point_vis = gaussian_pcd[max_indices].cpu().numpy().astype(np.float64)
    keypoints_matched = keypoints[..., :2].cpu().numpy().astype(np.float64)
    return point_vis, keypoints_matched


def get_coord(h=60, w=80):
    rows = torch.arange(0, h).cuda()
    cols = torch.arange(0, w).cuda()
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
    coords = torch.stack((grid_y, grid_x), dim=-1)
    coords_flat = coords.reshape(-1, 2)
    return coords_flat


def find_2d3d_dense(image_features, gaussian_pcd, gaussian_feat, chunk_size=10000):
    c, h, w = image_features.shape
    device = image_features.device

    tmp_coord = get_coord(h=h, w=w)
    tmp_feature = image_features.permute(1,2,0)
    tmp_feature = tmp_feature.reshape(-1, 256)
    f_N, feat_dim = tmp_feature.shape
    P_N = gaussian_feat.shape[0]

    tmp_feature = F.normalize(tmp_feature, p=2, dim=1)
    gaussian_feat = F.normalize(gaussian_feat, p=2, dim=1)

    max_similarity = torch.full((f_N,), -float('inf'), device=device)
    max_indices = torch.zeros(f_N, dtype=torch.long, device=device)
    for part in range(0, P_N, chunk_size):
        chunk = gaussian_feat[part:part + chunk_size]
        # Use matrix multiplication for faster similarity computation
        similarity = torch.mm(tmp_feature, chunk.t())
        chunk_max, chunk_indices = similarity.max(dim=1)
        update_mask = chunk_max > max_similarity
        max_similarity[update_mask] = chunk_max[update_mask]
        max_indices[update_mask] = chunk_indices[update_mask] + part
    final_mask = max_similarity>0.7
    max_indices = max_indices[final_mask]
    point_vis = gaussian_pcd[max_indices].cpu().numpy().astype(np.float64)
    pt_matched = tmp_coord[final_mask].cpu().numpy().astype(np.float64)
    pt_matched = pt_matched*8
    print(point_vis.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_vis)
    pcd.paint_uniform_color([0, 0, 0])
    o3d.io.write_point_cloud(f"pcd_match.ply", pcd)
    return point_vis, pt_matched




# filter gaussian based on opacity
def save_pcd(center_points, name=""):
    center_points = center_points.detach().cpu().numpy()
    center_pcd = o3d.geometry.PointCloud()
    center_pcd.points = o3d.utility.Vector3dVector(center_points)
    center_pcd.paint_uniform_color([1, 0, 0])  # Green for cluster centers
    o3d.io.write_point_cloud(f"centers_{name}.ply", center_pcd)


def find_gaussian_score_opa(gaussians):
    gaussian_pcd = gaussians.get_xyz
    gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
    scores = gaussians.get_score_feature.squeeze(-1)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    opacities = gaussians.get_opacity
    filter = opacities
    th0 = choose_th(filter, 0.95)
    mask0 = filter>th0
    mask0 = mask0.squeeze(-1)
    filtered_points = gaussian_pcd[mask0]
    filtered_feature = gaussian_feat[mask0]
    filtered_scores = scores[mask0]
    save_pcd(filtered_points, '0')
    print(filtered_points.shape)
    return filtered_points, filtered_feature





# other matchers
original_size = (480, 640)
def img_match_mast3r(img0, img1, model, K, depth_map, w2c, gt_pose_44):
    s=time.time()
    image1_tensor = img0
    image2_tensor = img1.unsqueeze(0)
    image1 = convert_tensor_to_dust3r_format(image1_tensor, size=512, idx=0)
    image2 = convert_tensor_to_dust3r_format(image2_tensor, size=512, idx=1)
    print(f"time 1: {time.time()-s} s")
    images = [image1, image2]
    output = inference([tuple(images)], model, device = 'cuda', batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    print(f"time 2: {time.time()-s} s")
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                device='cuda', dist='dot', block_size=2**13)
    print(f"time 3: {time.time()-s} s")
    H0, W0 = view1['true_shape'][0]
    
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    scale_x = original_size[1] / W0.item()
    scale_y = original_size[0] / H0.item()
    for pixel in matches_im1:
        pixel[0] *= scale_x
        pixel[1] *= scale_y
    for pixel in matches_im0:
        pixel[0] *= scale_x
        pixel[1] *= scale_y
    dist_eff = np.array([0,0,0,0], dtype=np.float32)

    predict_c2w_ini = np.linalg.inv(w2c.cpu().detach().numpy())
    predict_w2c_ini = w2c.cpu().detach().numpy()
    initial_rvec, _ = cv2.Rodrigues(predict_c2w_ini[:3,:3].astype(np.float32))
    initial_tvec = predict_c2w_ini[:3,3].astype(np.float32)
    gt_c2w_pose = gt_pose_44.cpu().detach().numpy()
    K_inv = np.linalg.inv(K)
    depth_map = depth_map.squeeze(0).cpu().detach().numpy()
    height, width = depth_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    depth_flat = depth_map.flatten()
    x_normalized = (x_flat - K[0, 2]) / K[0, 0]
    y_normalized = (y_flat - K[1, 2]) / K[1, 1]
    X_camera = depth_flat * x_normalized
    Y_camera = depth_flat * y_normalized
    Z_camera = depth_flat
    points_camera = np.vstack((X_camera, Y_camera, Z_camera, np.ones_like(X_camera)))
    points_world = predict_c2w_ini @ points_camera
    X_world = points_world[0, :]
    Y_world = points_world[1, :]
    Z_world = points_world[2, :]
    points_3D = np.vstack((X_world, Y_world, Z_world))
    scene_coordinates_gs = points_3D.reshape(3, original_size[0], original_size[1])
    points_3D_at_pixels = np.zeros((matches_im0.shape[0], 3))
    for i, (x, y) in enumerate(matches_im0):
        points_3D_at_pixels[i] = scene_coordinates_gs[:, y, x]
    
    print(f"time 4: {time.time()-s} s")
    if matches_im1.shape[0] >= 4:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D_at_pixels.astype(np.float32), 
                                                          matches_im1.astype(np.float32), 
                                                          K, dist_eff,
                                                          rvec=initial_rvec, tvec=initial_tvec, 
                                                          useExtrinsicGuess=True, reprojectionError=1.0,
                                                          iterationsCount=2000,flags=cv2.SOLVEPNP_EPNP)
        print(f"time 5: {time.time()-s} s")
        R = perform_rodrigues_transformation(rvec)
        trans = -R.T @ np.matrix(tvec)
        predict_c2w_refine = np.eye(4)
        predict_c2w_refine[:3,:3] = R.T
        predict_c2w_refine[:3,3] = trans.reshape(3)
        ini_rot_error,ini_translation_error=cal_campose_error(predict_c2w_ini, gt_c2w_pose)
        refine_rot_error, refine_translation_error=cal_campose_error(predict_c2w_refine, gt_c2w_pose)
        combined_list = rotmat2qvec(np.linalg.inv(predict_c2w_refine)[:3,:3]).tolist() + \
            np.linalg.inv(predict_c2w_refine)[:3,3].tolist()
        output_line = ' '.join(map(str, combined_list))
        print(f"time 6: {time.time()-s} s")
    return refine_rot_error, refine_translation_error


def img_match_loftr(img0, img1, matcher):
    img0 = rgb2loftrgray(img0.squeeze(0))
    img1 = rgb2loftrgray(img1)
    batch = {'image0':img0, 'image1': img1}
    matcher(batch)
    mkpts0 = batch['mkpts0_f']
    mkpts1 = batch['mkpts1_f']
    mconf = batch['mconf']
    batch["mkpt0"] = batch['mkpts0_f']
    batch["mkpt1"] = batch['mkpts1_f']
    batch["img0"] = (img0.cpu().detach().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
    batch["img1"] = (img1["render"].cpu().detach().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
    breakpoint()


def img_match_aspan(img0, img1, matcher):
    st = time.time()
    matcher_name = "ASpanFormer"
    with torch.no_grad():
        img0 = rgb2loftrgray(img0.squeeze(0))
        img1 = rgb2loftrgray(img1)
        data = {
            "img0": img0.cuda(),
            "img1": img1.cuda(),
        }
        print("time1:", time.time()-st)
        semi_img_match(data, matcher)
        print("time2:", time.time()-st)
    return data






def choose_th(feature, histogram_th):
    score_flat = feature.flatten()
    percentile_value = torch.quantile(score_flat, float(histogram_th))
    return percentile_value.item()

# loc inference
def img_match_ours(args, mlp, 
                   img0, 
                   scoremap1, featmap1, 
                   encoder,   matcher):
    tmp = {}
    tmp["image"] = img0
    tmp_pred = encoder(tmp)
    #######################################
    desc = tmp_pred["descriptors"]
    compressed_gt_feature = mlp(desc)
    gt_feature = mlp.decode(compressed_gt_feature)
    # gt_feature = desc
    #######################################
    kpt0 = tmp_pred["keypoints"]
    feat0 = gt_feature
    #######################################
    if args.kpt_hist is not None:
        kpt_th = choose_th(scoremap1, args.kpt_hist)
    else:
        kpt_th = args.kpt_th
    #######################################
    kpt1 = find_small_circle_centers(scoremap1, threshold=kpt_th, kernel_size=args.kernel_size)
    if kpt1.shape[0] == 0:
        return None
    kpt1 = kpt1.clone().detach()[:, [1, 0]].to(scoremap1)
    _, h, w = scoremap1.shape
    scale = torch.tensor([w, h]).to(scoremap1)
    #######################################
    feat1 = sample_descriptors_fix_sampling(kpt1, featmap1, scale)
    feat1 = mlp.decode(feat1)
    #######################################
    data = {}
    data["keypoints0"] = kpt0
    data["keypoints1"] = kpt1.unsqueeze(0)
    data["descriptors0"] = feat0
    data["descriptors1"] = feat1
    data["image_size"] = scoremap1.shape[1:]
    pred = matcher(data)
    #######################################
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = data["keypoints0"][0][valid].cpu(), data["keypoints1"][0][m0[0][valid]].cpu()
    result = {}
    result['mkpt0'] = m0
    result['mkpt1'] = m1
    result['kpt0'] = kpt0[0].cpu()
    result['kpt1'] = kpt1[0].cpu()
    return result


def img_match_rival(img0, img1, encoder, matcher) -> dict:
    d0 = {}
    d1 = {}
    d0['image'] = img0
    d1['image'] = img1.unsqueeze(0)
    p0 = encoder(d0)
    p1 = encoder(d1)
    ########################################################
    tmp = {}
    tmp["keypoints0"] = p0['keypoints']
    tmp["keypoints1"] = p1['keypoints']
    tmp["descriptors0"] = p0['descriptors']
    tmp["descriptors1"] = p1['descriptors']
    tmp["image_size"] = d0['image'][0].shape[1:]
    ########################################################
    if tmp["keypoints1"].shape[1]==0:
        return None
    pred = matcher(tmp)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[0][valid]].cpu()
    kpt0, kpt1 = tmp['keypoints0'][0].cpu(), tmp['keypoints1'][0].cpu()
    ########################################################
    result = {}
    result['mkpt0'] = m0
    result['mkpt1'] = m1
    result['kpt0'] = kpt0
    result['kpt1'] = kpt1
    return result


def img_match_kptSPfeat(args, 
                       img0,     img1,
                       scoremap1,
                       encoder, matcher):
    d0 = {}
    d1 = {}
    d0['image'] = img0
    d1['image'] = img1.unsqueeze(0)
    p0 = encoder(d0)
    p1 = encoder(d1)
    #######################################
    if args.kpt_hist is not None:
        kpt_th = choose_th(scoremap1, args.kpt_hist)
    else:
        kpt_th = args.kpt_th
    kpt1 = find_small_circle_centers(scoremap1, threshold=kpt_th, kernel_size=args.kernel_size)
    if kpt1.shape[0] == 0:
        return None
    kpt1 = kpt1.clone().detach()[:, [1, 0]].to(scoremap1)
    _, h, w = scoremap1.shape
    scale = torch.tensor([w, h]).to(scoremap1)
    featmap1 = p1["dense_descriptors"][0]
    feat1 = sample_descriptors_fix_sampling(kpt1, featmap1, scale)
    #######################################
    tmp = {}
    tmp["keypoints0"] = p0['keypoints']
    tmp["descriptors0"] = p0['descriptors']
    tmp["keypoints1"] = kpt1.unsqueeze(0)
    tmp["descriptors1"] = feat1
    tmp["image_size"] = d0['image'][0].shape[1:]
    if tmp["keypoints1"].shape[1]==0:
        return None
    #######################################
    pred = matcher(tmp)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[0][valid]].cpu()
    kpt0, kpt1 = tmp['keypoints0'][0].cpu(), tmp['keypoints1'][0].cpu()
    result = {}
    result['mkpt0'] = m0
    result['mkpt1'] = m1
    result['kpt0'] = kpt0
    result['kpt1'] = kpt1
    return result


def img_match_RenderRender(args, mlp, matcher,
                            scoremap0, featmap0,
                            scoremap1, featmap1,):
    th0 = choose_th(scoremap0, args.kpt_hist)
    th1 = choose_th(scoremap1, args.kpt_hist)
    kpt0 = find_small_circle_centers(scoremap0, threshold=th0, kernel_size=args.kernel_size).clone().detach()[:, [1, 0]]
    kpt1 = find_small_circle_centers(scoremap1, threshold=th1, kernel_size=args.kernel_size).clone().detach()[:, [1, 0]]
    _, h, w = scoremap0.shape
    scale = torch.tensor([w, h]).to(scoremap0)
    feat0 = sample_descriptors_fix_sampling(kpt0, featmap0, scale)
    feat1 = sample_descriptors_fix_sampling(kpt1, featmap1, scale)
    feat0 = mlp.decode(feat0)
    feat1 = mlp.decode(feat1)
    #######################################
    data = {}
    data["keypoints0"] = kpt0.unsqueeze(0)
    data["keypoints1"] = kpt1.unsqueeze(0)
    data["descriptors0"] = feat0
    data["descriptors1"] = feat1
    data["image_size"] = scoremap1.shape[1:]
    pred = matcher(data)
    #######################################
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = data["keypoints0"][0][valid].cpu(), data["keypoints1"][0][m0[0][valid]].cpu()
    result = {}
    result['mkpt0'] = m0
    result['mkpt1'] = m1
    result['kpt0'] = kpt0.cpu()
    result['kpt1'] = kpt1.cpu()
    return result
