import time
import torch
import numpy as np
from utils.loc.loc_utils import *
from argparse import ArgumentParser
from scene.gaussian.gaussian_model import GaussianModel
from scene import Scene
from utils.match.match_img import save_matchimg
from utils.graphics_utils import fov2focal
from utils.loc.depth import project_2d_to_3d
from gaussian_renderer import render_from_pose_gsplat
from matchers.lightglue import LightGlue
from encoders.superpoint.superpoint import SuperPoint
from arguments import ModelParams, PipelineParams, get_combined_args
from mlp.mlp import get_mlp_model, get_mlp_dataset, get_mlp_augment
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


def localize_set(model_path, name, views, gaussians, pipe_param, background, args, encoder, matcher):
    rErrs = []
    tErrs = []
    prior_rErr = []
    prior_tErr = []
    scene_name = model_path.split('/')[-3]
    if args.mlp_method.startswith("SP"):
        mlp = get_mlp_model(dim = args.mlp_dim, type=args.mlp_method)
    elif args.mlp_method.startswith("pgt") or args.mlp_method.startswith("pairs") or args.mlp_method.startswith("match"):
        mlp = get_mlp_dataset(dim=args.mlp_dim, dataset=args.mlp_method)
    elif args.mlp_method == "Cambridge":
        mlp = get_mlp_dataset(dim=args.mlp_dim, dataset=args.mlp_method)
    elif args.mlp_method.startswith("Cambridge"):
        mlp = get_mlp_dataset(dim=args.mlp_dim, dataset=args.mlp_method)
    elif args.mlp_method.startswith("augment"):
        mlp = get_mlp_augment(dim=args.mlp_dim, dataset=args.mlp_method)
    mlp = mlp.to("cuda").eval()
    print(scene_name)

    gaussian_pcd = gaussians.get_xyz
    gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
    gaussian_feat_h = mlp.decode(gaussian_feat)
    test_name = f"iteration{args.iteration}_sp{args.sp_th}_lg{args.lg_th}_kptth{args.kpt_th}_\
                    kpthist{args.kpt_hist}_ransaciters{args.ransac_iters}"
    if args.save_match:
        match_folder = f'{model_path}/match_imgs/{test_name}'
        os.makedirs(match_folder, exist_ok=True)

    # Create progress bar for localization
    pbar = tqdm(enumerate(views), total=len(views), desc="Localizing")
    
    for index, _ in pbar:
        view = views[index]
        start = time.time()
        gt_im = view.original_image[0:3, :, :].cuda().unsqueeze(0)
        data = {}
        data["image"] = gt_im
        pred = encoder(data)
        gt_keypoints = pred["keypoints"].squeeze(0)
        desc = pred["descriptors"].squeeze(0)
        dense_desc = pred["dense_descriptors"].squeeze(0)
        gt_feature = mlp(desc)

        K = np.eye(3)
        focal_length = fov2focal(view.FoVx, view.image_width)
        K[0, 0] = K[1, 1] = focal_length
        K[0, 2] = view.image_width / 2
        K[1, 2] = view.image_height / 2
        start = time.time()
        with torch.no_grad():
            matched_3d, matched_2d = find_2d3d_correspondences(gt_keypoints.detach(), gt_feature.detach(),
                                                                gaussian_pcd.detach(), gaussian_feat.detach())
        gt_R = view.R
        gt_t = view.T
        _, R, t, _ = cv2.solvePnPRansac(matched_3d, matched_2d, 
                                        K, 
                                        distCoeffs=None, 
                                        flags=cv2.SOLVEPNP_ITERATIVE, 
                                        iterationsCount=args.ransac_iters
                                        )
        R, _ = cv2.Rodrigues(R)
        rotError, transError = calculate_pose_errors(gt_R, gt_t, R.T, t)
        
        # Update progress bar with initial errors
        # pbar.set_postfix({
        #     'Image': view.image_name,
        #     'Initial_R_err': f'{rotError:.2f}°',
        #     'Initial_T_err': f'{transError:.2f}cm'
        # })

        # breakpoint()

        w2c = torch.eye(4, 4, device='cuda')
        w2c[:3, :3] = torch.from_numpy(R).float()
        w2c[:3, 3] = torch.from_numpy(t[:, 0]).float()
        view.update_RT(R.T, t[:,0])

        with torch.no_grad():
            # Extract pose matrix and camera parameters for render_from_pose_gsplat
            pose = view.world_view_transform.transpose(0, 1).cuda()  # Convert to world-to-camera matrix
            fovx = view.FoVx
            fovy = view.FoVy
            width = view.image_width
            height = view.image_height
            render_pkg = render_from_pose_gsplat(gaussians, pose, fovx, fovy, width, height, 
                                               bg_color=background, rgb_only=False)
        
        db_render = render_pkg["render"]
        db_score = render_pkg["score_map"]
        db_feature = render_pkg["feature_map"]
        db_depth = render_pkg["depth"]
        query_render = gt_im

        result = img_match_ours(args, query_render, db_score, db_feature, encoder, matcher, mlp)
        if result is None:
            prior_rErr.append(rotError)
            prior_tErr.append(transError)
            rErrs.append(rotError)
            tErrs.append(transError)
            pbar.set_postfix({
                #'Image': view.image_name,
                'I_AE': f'{rotError:.2f}°',
                'I_TE': f'{transError:.2f}cm',
                'Status': 'No matches'
            })
            continue
        if not len(result['mkpt1'].cpu())>4:
            prior_rErr.append(rotError)
            prior_tErr.append(transError)
            rErrs.append(rotError)
            tErrs.append(transError)
            pbar.set_postfix({
                #'Image': view.image_name,
                'AE': f'{rotError:.2f}°',
                'TE': f'{transError:.2f}cm',
                'Status': 'Insufficient matches'
            })
            continue

        result['img0'] = gt_im.squeeze(0).permute(1, 2, 0)
        result['img1'] = db_render.squeeze(0).permute(1, 2, 0)
        
        if args.save_match:
            save_matchimg(result, f'{match_folder}/{index}_{view.image_name}.png')

        db_world = project_2d_to_3d(result['mkpt1'].cpu(), db_depth.cpu(), torch.tensor(K, dtype=torch.float32).cpu(), 
                                    w2c.cpu()).cpu().numpy().astype(np.float64)
        q_matched = result['mkpt0'].cpu().numpy().astype(np.float64)
        _, R_final, t_final, _ = cv2.solvePnPRansac(db_world, q_matched, K, distCoeffs=None, 
                                                    flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=args.ransac_iters)
        R_final, _ = cv2.Rodrigues(R_final)
        rotError_final, transError_final = calculate_pose_errors(gt_R, gt_t, R_final.T, t_final)

        # Update progress bar with final errors
        elapsed_time = time.time() - start
        pbar.set_postfix({
            # 'Image': view.image_name,                
            'I_AE': f'{rotError:.2f}°',
            'I_TE': f'{transError:.2f}cm',
            'F_AE': f'{rotError_final:.2f}°',
            'F_TE': f'{transError_final:.2f}cm',
            'Time': f'{elapsed_time:.2f}s'
        })
        
        prior_rErr.append(rotError)
        prior_tErr.append(transError)
        rErrs.append(rotError_final)
        tErrs.append(transError_final)

    # Close progress bar
    pbar.close()

    error_foler = f'{model_path}/error_logs/{test_name}'
    os.makedirs(error_foler, exist_ok=True)
    print(f'\nLocalization complete!')
    print(f'Images processed: {len(prior_rErr)}')
    print(f'Successful localizations: {len(rErrs)}')
    print()
    log_errors(error_foler, prior_rErr, prior_tErr, list_text="", error_text=f"prior")
    log_errors(error_foler, rErrs, tErrs, list_text="", error_text="warp")

def plot_pose_error_analysis(split_errors, scene_name, output_file):
    """Plot histogram and scatter plot analysis of pose errors, color-coded by split."""
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    colors = {'train': 'tab:blue', 'test': 'tab:orange'}
    # Translation error histogram
    ax1 = fig.add_subplot(gs[0, 0])
    for split, (t_err, _) in split_errors.items():
        ax1.hist(t_err, bins=50, alpha=0.6, color=colors[split], edgecolor='black', label=f'{split.capitalize()}')
        ax1.axvline(np.mean(t_err), color=colors[split], linestyle='--', linewidth=2, label=f'{split.capitalize()} Mean: {np.mean(t_err):.3f}m')
        ax1.axvline(np.median(t_err), color=colors[split], linestyle=':', linewidth=2, label=f'{split.capitalize()} Median: {np.median(t_err):.3f}m')
    ax1.set_xlabel('Translation Error (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Translation Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Rotation error histogram
    ax2 = fig.add_subplot(gs[0, 1])
    for split, (_, r_err) in split_errors.items():
        ax2.hist(r_err, bins=50, alpha=0.6, color=colors[split], edgecolor='black', label=f'{split.capitalize()}')
        ax2.axvline(np.mean(r_err), color=colors[split], linestyle='--', linewidth=2, label=f'{split.capitalize()} Mean: {np.mean(r_err):.3f}°')
        ax2.axvline(np.median(r_err), color=colors[split], linestyle=':', linewidth=2, label=f'{split.capitalize()} Median: {np.median(r_err):.3f}°')
    ax2.set_xlabel('Rotation Error (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Rotation Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Translation vs Rotation scatter plot
    ax3 = fig.add_subplot(gs[0, 2])
    for split, (t_err, r_err) in split_errors.items():
        ax3.scatter(t_err, r_err, alpha=0.6, color=colors[split], s=10, label=f'{split.capitalize()}')
    ax3.set_xlabel('Translation Error (m)')
    ax3.set_ylabel('Rotation Error (degrees)')
    ax3.set_title('Translation vs Rotation Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.suptitle(f'Pose Error Analysis - {scene_name.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pose error analysis plot saved to: {output_file}")

def plot_success_rate_curves(split_errors, scene_name, output_file):
    """Plot success rate curves vs error thresholds, color-coded by split."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'train': 'tab:blue', 'test': 'tab:orange'}
    # Translation success rate curve
    t_thresholds = np.linspace(0, 2.5, 100)
    for split, (t_err, _) in split_errors.items():
        t_success_rates = [np.mean(t_err <= threshold) * 100 for threshold in t_thresholds]
        ax1.plot(t_thresholds, t_success_rates, linewidth=2, color=colors[split], label=f'{split.capitalize()}')
    ax1.set_xlabel('Translation Error Threshold (m)')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Translation Success Rate vs Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 100)
    ax1.legend()
    # Rotation success rate curve
    r_thresholds = np.linspace(0, 90, 100)
    for split, (_, r_err) in split_errors.items():
        r_success_rates = [np.mean(r_err <= threshold) * 100 for threshold in r_thresholds]
        ax2.plot(r_thresholds, r_success_rates, linewidth=2, color=colors[split], label=f'{split.capitalize()}')
    ax2.set_xlabel('Rotation Error Threshold (degrees)')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Rotation Success Rate vs Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 100)
    ax2.legend()
    plt.suptitle(f'Success Rate Curves - {scene_name.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Success rate curves plot saved to: {output_file}")


def localize(model_param:ModelParams, pipe_param:PipelineParams, args):
    gaussians = GaussianModel(model_param.sh_degree)
    scene = Scene(model_param, gaussians, load_iteration=args.iteration, shuffle=False, load_feature=False)
    bg_color = [1,1,1] if model_param.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": 1024,
        "detection_threshold": args.sp_th,
    }
    encoder = SuperPoint(conf).cuda().eval()
    matcher = LightGlue({"filter_threshold": args.lg_th ,}).cuda().eval()
    localize_set(model_param.model_path, "test", scene.getTestCameras(), 
                 gaussians, pipe_param, background, args, encoder, matcher)




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ransac_iters", default=20000, type=int)
    # parser.add_argument("--mlp_dim", required=True, type=int)
    parser.add_argument("--mlp_method", required=True, type=str)
    parser.add_argument("--save_match", action='store_true', help='Save match if this flag is provided.')
    parser.add_argument("--sp_th", default=0.01, type=float)
    parser.add_argument("--lg_th", default=0.01, type=float)
    parser.add_argument("--kpt_th", default=0.01, type=float)
    parser.add_argument("--kpt_hist", default=0.9, type=float)
    # kernel_size
    parser.add_argument("--kernel_size", default=13, type=int)
    args = get_combined_args(parser)
    localize(Model_param.extract(args), Pipe_param.extract(args), args)
