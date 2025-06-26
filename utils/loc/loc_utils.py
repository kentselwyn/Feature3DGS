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
from utils.match.match_img import extract_kpt, sample_descriptors_fix_sampling, save_matchimg, find_small_circle_centers
from matchers.mast3r.mast3r.fast_nn import fast_reciprocal_NNs
from matchers.mast3r.dust3r.dust3r.inference import inference
from matchers.mast3r.dust3r.dust3r.utils.image import convert_tensor_to_dust3r_format
import torchvision.transforms as T
from matchers.mast3r.utils.functions import *
from utils.match.match_img import semi_img_match


def calculate_pose_errors(R_gt, t_gt, R_est, t_est):
    # Calculate rotation error
    rotError = np.matmul(R_est.T, R_gt)
    rotError = cv2.Rodrigues(rotError)[0]
    rotError = np.linalg.norm(rotError) * 180 / np.pi
    # Calculate translation error
    transError = np.linalg.norm(t_gt - t_est.squeeze(1)) * 100  # Convert to cm
    return rotError, transError


def calculate_pose_errors2(R_gt, t_gt, R_est, t_est):
    # Calculate rotation error
    rotError = np.matmul(R_est.T, R_gt)
    rotError = cv2.Rodrigues(rotError)[0]
    rotError = np.linalg.norm(rotError) * 180 / np.pi
    # Calculate translation error
    transError = np.linalg.norm(t_gt - t_est) * 100  # Convert to cm
    return rotError, transError


def calculate_pose_errors_ace(gt_pose_44, out_pose):
    # Calculate translation error.
    t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))*100
    # Rotation error.
    gt_R = gt_pose_44[0:3, 0:3].numpy()
    out_R = out_pose[0:3, 0:3].numpy()
    r_err = np.matmul(out_R, np.transpose(gt_R))
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi
    return r_err, t_err



def log_errors(log_dir, name, rotation_errors, translation_errors, list_text, error_text, elapsed_time=None):
    total_frames = len(rotation_errors)
    # Remove NaN values from rotation_errors and translation_errors
    rotation_errors = [err for err in rotation_errors if not np.isnan(err)]
    translation_errors = [err for err in translation_errors if not np.isnan(err)]

    # Ensure both lists have the same length after NaN removal
    min_length = min(len(rotation_errors), len(translation_errors))
    rotation_errors = rotation_errors[:min_length]
    translation_errors = translation_errors[:min_length]

    # Update total_frames after NaN removal
    total_frames = len(rotation_errors)
    median_rErr = np.median(rotation_errors)
    median_tErr = np.median(translation_errors)

    # Compute accuracy percentages
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

    # Log median errors to separate files
    # log_dir = os.path.join(model_path, 'error_log_f_3')
    # os.makedirs(log_dir, exist_ok=True)
    error_list = rotation_errors + translation_errors
    with open(os.path.join(log_dir, 
                           f'error_list_{name}_{list_text}.pickle'), 'wb') as f:
        pickle.dump(error_list, f)
    with open(os.path.join(log_dir, 
                           f'median_error_{name}_{error_text}_end.txt'), 'a') as f:
        f.write('Accuracy:\n')
        f.write(f'\t10cm/5deg: {pct10_5:.1f}%\n')
        f.write(f'\t5cm/5deg: {pct5:.1f}%\n')
        f.write(f'\t2cm/2deg: {pct2:.1f}%\n')
        f.write(f'\t1cm/1deg: {pct1:.1f}%\n')
        f.write(f'Median translation error: {median_tErr:.6f} cm\n')
        f.write(f'Median rotation error: {median_rErr:.6f} dg\n')
        if elapsed_time is not None:
            f.write(f'Mean elapsed time: {elapsed_time:.6f} s\n')



def log_errors_iters(model_path, name, rotation_errors, translation_errors, inplace_text):
    
    # Remove NaN values from rotation_errors and translation_errors
    rotation_errors = {iter_num: [err for err in errors if not np.isnan(err)] for iter_num, errors in rotation_errors.items()}
    translation_errors = {iter_num: [err for err in errors if not np.isnan(err)] for iter_num, errors in translation_errors.items()}

    log_dir = os.path.join(model_path, 'error_logs')
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, f'median_error_{name}_{inplace_text}_iters_end.txt'), 'w') as f:
        for iter_num in sorted(rotation_errors.keys()):
            median_rErr = np.median(rotation_errors[iter_num])
            median_tErr = np.median(translation_errors[iter_num])
            f.write(f'{iter_num} iter | t_err: {median_tErr:.6f} cm, r_err: {median_rErr:.6f} deg\n')



def find_2d3d_correspondences(keypoints, image_features, gaussian_pcd, gaussian_feat, chunk_size=10000):
    device = image_features.device
    f_N, feat_dim = image_features.shape
    P_N = gaussian_feat.shape[0]
    
    # Normalize features for faster cosine similarity computation
    image_features = F.normalize(image_features, p=2, dim=1)
    gaussian_feat = F.normalize(gaussian_feat, p=2, dim=1)
    
    max_similarity = torch.full((f_N,), -float('inf'), device=device)
    max_indices = torch.zeros(f_N, dtype=torch.long, device=device)
    
    for part in range(0, P_N, chunk_size):
        chunk = gaussian_feat[part:part + chunk_size]
        # Use matrix multiplication for faster similarity computation
        similarity = torch.mm(image_features, chunk.t())
        
        chunk_max, chunk_indices = similarity.max(dim=1)
        update_mask = chunk_max > max_similarity
        max_similarity[update_mask] = chunk_max[update_mask]
        max_indices[update_mask] = chunk_indices[update_mask] + part
    # print(max_similarity)
    
    # final_mask = max_similarity>0.8
    # max_indices = max_indices[final_mask]
    # keypoints = keypoints[final_mask]
    point_vis = gaussian_pcd[max_indices].cpu().numpy().astype(np.float64)
    keypoints_matched = keypoints[..., :2].cpu().numpy().astype(np.float64)
    # print(point_vis.shape)


    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_vis)
    # pcd.paint_uniform_color([0, 0, 0])
    # o3d.io.write_point_cloud(f"pcd_match_0_.85.ply", pcd)
    
    return point_vis, keypoints_matched


def get_coord(h=60, w=80):
    rows = torch.arange(0, h).cuda()
    cols = torch.arange(0, w).cuda()
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
    coords = torch.stack((grid_y, grid_x), dim=-1)
    coords_flat = coords.reshape(-1, 2)
    return coords_flat


def find_2d3d_dense(image_features, gaussian_pcd, gaussian_feat, chunk_size=10000):
    # print(image_features.shape)
    # print(gaussian_pcd.shape)
    # print(gaussian_feat.shape)
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
    # print(max_similarity)
    final_mask = max_similarity>0.7
    max_indices = max_indices[final_mask]

    point_vis = gaussian_pcd[max_indices].cpu().numpy().astype(np.float64)
    pt_matched = tmp_coord[final_mask].cpu().numpy().astype(np.float64)
    pt_matched = pt_matched*8
    # keypoints_matched = keypoints[..., :2].cpu().numpy().astype(np.float64)
    print(point_vis.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_vis)
    pcd.paint_uniform_color([0, 0, 0])
    o3d.io.write_point_cloud(f"pcd_match.ply", pcd)

    return point_vis, pt_matched




def choose_th(feature, histogram_th):
    score_flat = feature.flatten()
    percentile_value = torch.quantile(score_flat, float(histogram_th))
    return percentile_value.item()


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


def find_2d3d_lg(img_kpts, img_feat, gs_pts, gs_feat, matcher, width, height, gt_img, index):
    # print(img_kpts)
    print(img_kpts.shape)
    # print(gs_pts)
    print(gs_pts.shape)
    gs_2ds = gs_pts[:, :2]
    min_vals, _ = gs_2ds.min(dim=0, keepdim=True)
    max_vals, _ = gs_2ds.max(dim=0, keepdim=True)
    gs_2ds_norm = (gs_2ds - min_vals) / (max_vals - min_vals)

    data = {}
    # test_mask = gs_2ds_norm[:, 0]<0.45
    # gs_2ds_norm = gs_2ds_norm[test_mask]
    # gs_feat = gs_feat[test_mask]
    size = torch.tensor([height, width], device="cuda")
    data["keypoints0"] = img_kpts.unsqueeze(0)
    data["keypoints1"] = (gs_2ds_norm * size).unsqueeze(0)
    data["descriptors0"] = img_feat.unsqueeze(0)
    data["descriptors1"] = gs_feat.unsqueeze(0)
    data["image_size"] = size

    pred = matcher(data)

    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = data["keypoints0"][0][valid].cpu(), data["keypoints1"][0][m0[0][valid]].cpu()
    result = {}
    result['mkpt0'] = m0
    result['mkpt1'] = m1
    result['kpt0'] = img_kpts.cpu()
    result['kpt1'] = (gs_2ds_norm * size).cpu()
    result['img0'] = gt_img.squeeze(0).permute(1, 2, 0)
    result['img1'] = torch.zeros([480, 640, 3], device="cuda")

    # result['img1'] = db_render.squeeze(0).permute(1, 2, 0)
    save_matchimg(result, f'2d3d_match_{index}.png')



def match_img(render_q, score_db, feature_db, encoder, matcher, mlp, args):
    tmp = {}
    tmp["image"] = render_q
    stt = time.time()
    tmp_pred = encoder(tmp)
    x = time.time()
    print("sp time: ",x-stt)
    desc = tmp_pred["descriptors"]
    compressed_gt_feature = mlp(desc)
    gt_feature = mlp.decode(compressed_gt_feature)
    
    # query
    kpt_q = tmp_pred["keypoints"]
    feat_q = gt_feature
    # db
    print("mlp time: ",time.time()-x)
    x = time.time()
    if args.kpt_hist is not None:
        kpt_th = choose_th(score_db, args.kpt_hist)
    else:
        kpt_th = args.kpt_th
    # kpt_db = extract_kpt(score_db, threshold=kpt_th)
    kpt_db = find_small_circle_centers(score_db, threshold=kpt_th, kernel_size=args.kernel_size)
    if kpt_db.shape[0] == 0:
        return None
    print("find small circle time: ", time.time()-x)
    x = time.time()
    kpt_db = kpt_db.clone().detach()[:, [1, 0]].to(score_db)
    _, h, w = score_db.shape
    scale = torch.tensor([w, h]).to(score_db)
    feat_db = sample_descriptors_fix_sampling(kpt_db, feature_db, scale)
    feat_db = mlp.decode(feat_db)
    kpt_db = kpt_db.unsqueeze(0)
    # match
    print("sample descriptor time: ", time.time()-x)
    x = time.time()
    data = {}
    data["keypoints0"] = kpt_q
    data["keypoints1"] = kpt_db
    data["descriptors0"] = feat_q
    data["descriptors1"] = feat_db
    data["image_size"] = score_db.shape[1:]
    # breakpoint()
    pred = matcher(data)
    print("match time: ", time.time()-x)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = data["keypoints0"][0][valid].cpu(), data["keypoints1"][0][m0[0][valid]].cpu()
    result = {}
    result['mkpt0'] = m0
    result['mkpt1'] = m1
    result['kpt0'] = kpt_q[0].cpu()
    result['kpt1'] = kpt_db[0].cpu()
    return result





def match_img_feature(render_q, score_db, feature_db, encoder, matcher, mlp, args):
    tmp = {}
    tmp["image"] = render_q
    stt = time.time()
    tmp_pred = encoder(tmp)
    x = time.time()
    print("sp time: ",x-stt)
    desc = tmp_pred["descriptors"]
    compressed_gt_feature = mlp(desc)
    gt_feature = mlp.decode(compressed_gt_feature)
    # breakpoint()
    dense_desc = tmp_pred["dense_descriptors"].squeeze(0).permute(1,2,0)
    compressed_dense_desc = mlp(dense_desc)
    decoded_dense_desc = mlp.decode(compressed_dense_desc).permute(2,0,1)
    compressed_dense_desc = compressed_dense_desc.permute(2,0,1)
    # query
    kpt_q = tmp_pred["keypoints"]
    feat_q = gt_feature
    # db
    print("mlp time: ",time.time()-x)
    x = time.time()
    if args.kpt_hist is not None:
        kpt_th = choose_th(score_db, args.kpt_hist)
    else:
        kpt_th = args.kpt_th
    # kpt_db = extract_kpt(score_db, threshold=kpt_th)
    kpt_db = find_small_circle_centers(score_db, threshold=kpt_th, kernel_size=args.kernel_size)
    if kpt_db.shape[0] == 0:
        return None
    print("find small circle time: ", time.time()-x)
    x = time.time()
    kpt_db = kpt_db.clone().detach()[:, [1, 0]].to(score_db)
    _, h, w = score_db.shape
    scale = torch.tensor([w, h]).to(score_db)
    feat_db = sample_descriptors_fix_sampling(kpt_db, feature_db, scale)
    feat_db = mlp.decode(feat_db)
    kpt_db = kpt_db.unsqueeze(0)
    # match
    print("sample descriptor time: ", time.time()-x)
    x = time.time()
    data = {}
    data["keypoints0"] = kpt_q
    data["keypoints1"] = kpt_db
    data["descriptors0"] = feat_q
    data["descriptors1"] = feat_db
    data["image_size"] = score_db.shape[1:]
    pred = matcher(data)
    print("match time: ", time.time()-x)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = data["keypoints0"][0][valid].cpu(), data["keypoints1"][0][m0[0][valid]].cpu()
    result = {}
    result['mkpt0'] = m0
    result['mkpt1'] = m1
    result['kpt0'] = kpt_q[0].cpu()
    result['kpt1'] = kpt_db[0].cpu()
    return result, compressed_dense_desc, decoded_dense_desc


def img_match2(query_render, db_render, encoder, matcher) -> Tuple[torch.Tensor, torch.Tensor]:
    d0 = {}
    d1 = {}
    d0['image'] = query_render
    d1['image'] = db_render.unsqueeze(0)
    p0 = encoder(d0)
    p1 = encoder(d1)

    tmp = {}
    tmp["keypoints0"] = p0['keypoints']
    tmp["keypoints1"] = p1['keypoints']
    tmp["descriptors0"] = p0['descriptors']
    tmp["descriptors1"] = p1['descriptors']
    tmp["image_size"] = d0['image'][0].shape[1:]

    if tmp["keypoints1"].shape[1]==0:
        return None
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


def feat_fromImg_match(query_render, score_db, db_render, encoder, matcher, args):
    d0 = {}
    d1 = {}
    d0['image'] = query_render
    d1['image'] = db_render.unsqueeze(0)

    p0 = encoder(d0)
    p1 = encoder(d1)

    tmp = {}
    tmp["keypoints0"] = p0['keypoints']
    tmp["descriptors0"] = p0['descriptors']
    tmp["image_size"] = d0['image'][0].shape[1:]

    feature_db = p1["dense_descriptors"][0]
    if args.kpt_hist is not None:
        kpt_th = choose_th(score_db, args.kpt_hist)
    else:
        kpt_th = args.kpt_th
    kpt_db = find_small_circle_centers(score_db, threshold=kpt_th, kernel_size=args.kernel_size)
    if kpt_db.shape[0] == 0:
        return None

    kpt_db = kpt_db.clone().detach()[:, [1, 0]].to(score_db)
    _, h, w = score_db.shape
    scale = torch.tensor([w, h]).to(score_db)
    feat_db = sample_descriptors_fix_sampling(kpt_db, feature_db, scale)
    tmp["keypoints1"] = kpt_db.unsqueeze(0)
    tmp["descriptors1"] = feat_db

    if tmp["keypoints1"].shape[1]==0:
        return None
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
    

original_size = (480, 640)
def img_match_mast3r(query_render, db_render, model, K, depth_map, w2c, gt_pose_44):
    s=time.time()
    image1_tensor = query_render
    image2_tensor = db_render.unsqueeze(0)
    image1 = convert_tensor_to_dust3r_format(image1_tensor, size=512, idx=0)
    image2 = convert_tensor_to_dust3r_format(image2_tensor, size=512, idx=1)
    print(f"time 1: {time.time()-s} s")
    
    images = [image1, image2]
    output = inference([tuple(images)], model, device = 'cuda', batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    print(f"time 2: {time.time()-s} s")

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                device='cuda', dist='dot', block_size=2**13)
    
    print(f"time 3: {time.time()-s} s")

    # ignore small border around the edge
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
        refine_rot_error,refine_translation_error=cal_campose_error(predict_c2w_refine, gt_c2w_pose)

        combined_list = rotmat2qvec(np.linalg.inv(predict_c2w_refine)[:3,:3]).tolist() + \
            np.linalg.inv(predict_c2w_refine)[:3,3].tolist()
        output_line = ' '.join(map(str, combined_list))
        # print(output_line)
        print(f"time 6: {time.time()-s} s")
        # print(f"translation error: {refine_translation_error} cm")
        # print(f"rotation error: {refine_rot_error} deg")
    return refine_rot_error, refine_translation_error


def img_match_loftr(query_render, db_render, matcher):
    img0 = rgb2loftrgray(query_render.squeeze(0))
    img1 = rgb2loftrgray(db_render)
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


def img_match_aspan(query_render, db_render, matcher):
    st = time.time()
    matcher_name = "ASpanFormer"
    with torch.no_grad():
        img0 = rgb2loftrgray(query_render.squeeze(0))
        img1 = rgb2loftrgray(db_render)
        data = {
            "img0": img0.cuda(),
            "img1": img1.cuda(),
        }
        print("time1:", time.time()-st)
        semi_img_match(data, matcher)
        print("time2:", time.time()-st)
    return data
