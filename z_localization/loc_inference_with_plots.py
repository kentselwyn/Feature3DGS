import time
import torch
import numpy as np
import os
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
    print(f"Processing {name} set for {scene_name}")

    gaussian_pcd = gaussians.get_xyz
    gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
    gaussian_feat_h = mlp.decode(gaussian_feat)
    test_name = f"iteration{args.iteration}_sp{args.sp_th}_lg{args.lg_th}_kptth{args.kpt_th}_\
                    kpthist{args.kpt_hist}_ransaciters{args.ransac_iters}"
    if args.save_match:
        match_folder = f'{model_path}/match_imgs/{test_name}'
        os.makedirs(match_folder, exist_ok=True)

    # Create progress bar for localization
    pbar = tqdm(enumerate(views), total=len(views), desc=f"Localizing {name} set")
    
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
    print(f'\nLocalization complete for {name} set!')
    print(f'Images processed: {len(prior_rErr)}')
    print(f'Successful localizations: {len(rErrs)}')
    print()
    log_errors(error_foler, prior_rErr, prior_tErr, list_text="", error_text=f"prior_{name}")
    log_errors(error_foler, rErrs, tErrs, list_text="", error_text=f"warp_{name}")
    
    # Convert errors to meters and return for plotting
    tErrs_m = [t/100.0 for t in tErrs]  # Convert cm to meters
    prior_tErrs_m = [t/100.0 for t in prior_tErr]  # Convert cm to meters
    
    return {
        'translation_errors': tErrs_m,
        'rotation_errors': rErrs,
        'prior_translation_errors': prior_tErrs_m, 
        'prior_rotation_errors': prior_rErr,
        'scene_name': scene_name
    }

def plot_pose_error_analysis(split_errors, scene_name, output_file):
    """Plot histogram and scatter plot analysis of pose errors, color-coded by split."""
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    colors = {'train': 'tab:blue', 'test': 'tab:orange'}
    
    # Translation error histogram
    ax1 = fig.add_subplot(gs[0, 0])
    for split, (t_err, _) in split_errors.items():
        if len(t_err) > 0:
            ax1.hist(t_err, bins=50, alpha=0.6, color=colors[split], edgecolor='black', label=f'{split.capitalize()}')
            ax1.axvline(np.mean(t_err), color=colors[split], linestyle='--', linewidth=2, 
                       label=f'{split.capitalize()} Mean: {np.mean(t_err):.3f}m')
            ax1.axvline(np.median(t_err), color=colors[split], linestyle=':', linewidth=2, 
                       label=f'{split.capitalize()} Median: {np.median(t_err):.3f}m')
    ax1.set_xlabel('Translation Error (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Translation Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotation error histogram
    ax2 = fig.add_subplot(gs[0, 1])
    for split, (_, r_err) in split_errors.items():
        if len(r_err) > 0:
            ax2.hist(r_err, bins=50, alpha=0.6, color=colors[split], edgecolor='black', label=f'{split.capitalize()}')
            ax2.axvline(np.mean(r_err), color=colors[split], linestyle='--', linewidth=2, 
                       label=f'{split.capitalize()} Mean: {np.mean(r_err):.3f}°')
            ax2.axvline(np.median(r_err), color=colors[split], linestyle=':', linewidth=2, 
                       label=f'{split.capitalize()} Median: {np.median(r_err):.3f}°')
    ax2.set_xlabel('Rotation Error (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Rotation Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Translation vs Rotation scatter plot
    ax3 = fig.add_subplot(gs[0, 2])
    for split, (t_err, r_err) in split_errors.items():
        if len(t_err) > 0 and len(r_err) > 0:
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
        if len(t_err) > 0:
            t_err_array = np.array(t_err)
            t_success_rates = [np.mean(t_err_array <= threshold) * 100 for threshold in t_thresholds]
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
        if len(r_err) > 0:
            r_err_array = np.array(r_err)
            r_success_rates = [np.mean(r_err_array <= threshold) * 100 for threshold in r_thresholds]
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
    
    # Process only test set
    test_results = localize_set(model_param.model_path, "test", scene.getTestCameras(), 
                               gaussians, pipe_param, background, args, encoder, matcher)
    
    # Prepare data for plotting functions
    split_errors = {
        'test': (test_results['translation_errors'], test_results['rotation_errors']),
    }
    
    scene_name = test_results['scene_name']
    
    # Create output directory for plots
    test_name = f"iteration{args.iteration}_sp{args.sp_th}_lg{args.lg_th}_kptth{args.kpt_th}_\
                    kpthist{args.kpt_hist}_ransaciters{args.ransac_iters}"
    plot_folder = f'{model_param.model_path}/plots/{test_name}'
    os.makedirs(plot_folder, exist_ok=True)
    
    # Generate plots using the existing functions
    pose_analysis_file = f'{plot_folder}/pose_error_analysis_{scene_name}.png'
    success_curves_file = f'{plot_folder}/success_rate_curves_{scene_name}.png'
    
    plot_pose_error_analysis(split_errors, scene_name, pose_analysis_file)
    plot_success_rate_curves(split_errors, scene_name, success_curves_file)
    
    print(f"\nPlots saved to: {plot_folder}")
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    for split_name, (t_errors, r_errors) in split_errors.items():
        if len(t_errors) > 0:
            print(f"\n{split_name.upper()} SET:")
            print(f"  Images processed: {len(t_errors)}")
            print(f"  Translation Error - Mean: {np.mean(t_errors):.3f}m, Median: {np.median(t_errors):.3f}m")
            print(f"  Rotation Error - Mean: {np.mean(r_errors):.3f}°, Median: {np.median(r_errors):.3f}°")
            
            # Calculate success rates at different thresholds
            t_arr = np.array(t_errors)
            r_arr = np.array(r_errors)
            
            # Standard thresholds
            success_25cm_5deg = np.mean((t_arr <= 0.25) & (r_arr <= 5.0)) * 100
            success_50cm_10deg = np.mean((t_arr <= 0.50) & (r_arr <= 10.0)) * 100
            success_1m_15deg = np.mean((t_arr <= 1.0) & (r_arr <= 15.0)) * 100
            
            print(f"  Success Rate (0.25m, 5°): {success_25cm_5deg:.1f}%")
            print(f"  Success Rate (0.50m, 10°): {success_50cm_10deg:.1f}%")
            print(f"  Success Rate (1.0m, 15°): {success_1m_15deg:.1f}%")
        else:
            print(f"\n{split_name.upper()} SET: No successful localizations")

def create_combined_analysis_plot(split_errors, scene_name, output_file):
    """Create a comprehensive analysis plot with multiple subplots."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    colors = {'train': 'tab:blue', 'test': 'tab:orange'}
    
    # Translation error histograms
    ax1 = fig.add_subplot(gs[0, 0])
    for split, (t_err, _) in split_errors.items():
        if len(t_err) > 0:
            ax1.hist(t_err, bins=30, alpha=0.6, color=colors[split], edgecolor='black', label=f'{split.capitalize()}')
    ax1.set_xlabel('Translation Error (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Translation Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotation error histograms
    ax2 = fig.add_subplot(gs[0, 1])
    for split, (_, r_err) in split_errors.items():
        if len(r_err) > 0:
            ax2.hist(r_err, bins=30, alpha=0.6, color=colors[split], edgecolor='black', label=f'{split.capitalize()}')
    ax2.set_xlabel('Rotation Error (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Rotation Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Translation success rate curves
    ax3 = fig.add_subplot(gs[0, 2])
    t_thresholds = np.linspace(0, 2.0, 100)
    for split, (t_err, _) in split_errors.items():
        if len(t_err) > 0:
            t_err_array = np.array(t_err)
            t_success_rates = [np.mean(t_err_array <= threshold) * 100 for threshold in t_thresholds]
            ax3.plot(t_thresholds, t_success_rates, linewidth=2, color=colors[split], label=f'{split.capitalize()}')
    ax3.set_xlabel('Translation Threshold (m)')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Translation Success Rate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Rotation success rate curves
    ax4 = fig.add_subplot(gs[0, 3])
    r_thresholds = np.linspace(0, 45, 100)
    for split, (_, r_err) in split_errors.items():
        if len(r_err) > 0:
            r_err_array = np.array(r_err)
            r_success_rates = [np.mean(r_err_array <= threshold) * 100 for threshold in r_thresholds]
            ax4.plot(r_thresholds, r_success_rates, linewidth=2, color=colors[split], label=f'{split.capitalize()}')
    ax4.set_xlabel('Rotation Threshold (degrees)')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Rotation Success Rate')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Scatter plots
    ax5 = fig.add_subplot(gs[1, 0])
    for split, (t_err, r_err) in split_errors.items():
        if len(t_err) > 0 and len(r_err) > 0:
            ax5.scatter(t_err, r_err, alpha=0.6, color=colors[split], s=15, label=f'{split.capitalize()}')
    ax5.set_xlabel('Translation Error (m)')
    ax5.set_ylabel('Rotation Error (degrees)')
    ax5.set_title('Translation vs Rotation Error')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Box plots for translation errors
    ax6 = fig.add_subplot(gs[1, 1])
    t_data = [split_errors[split][0] for split in ['train', 'test'] if len(split_errors[split][0]) > 0]
    t_labels = [split.capitalize() for split in ['train', 'test'] if len(split_errors[split][0]) > 0]
    if t_data:
        bp1 = ax6.boxplot(t_data, labels=t_labels, patch_artist=True)
        for patch, color in zip(bp1['boxes'], [colors['train'], colors['test']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax6.set_ylabel('Translation Error (m)')
    ax6.set_title('Translation Error Box Plot')
    ax6.grid(True, alpha=0.3)
    
    # Box plots for rotation errors
    ax7 = fig.add_subplot(gs[1, 2])
    r_data = [split_errors[split][1] for split in ['train', 'test'] if len(split_errors[split][1]) > 0]
    r_labels = [split.capitalize() for split in ['train', 'test'] if len(split_errors[split][1]) > 0]
    if r_data:
        bp2 = ax7.boxplot(r_data, labels=r_labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], [colors['train'], colors['test']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax7.set_ylabel('Rotation Error (degrees)')
    ax7.set_title('Rotation Error Box Plot')
    ax7.grid(True, alpha=0.3)
    
    # Combined success rate (both translation and rotation thresholds)
    ax8 = fig.add_subplot(gs[1, 3])
    t_thresholds_combined = [0.1, 0.25, 0.5, 1.0, 2.0]
    r_thresholds_combined = [2, 5, 10, 15, 30]
    
    x_pos = np.arange(len(t_thresholds_combined))
    width = 0.35
    
    for i, split in enumerate(['train', 'test']):
        t_err, r_err = split_errors[split]
        if len(t_err) > 0 and len(r_err) > 0:
            t_arr = np.array(t_err)
            r_arr = np.array(r_err)
            success_rates = []
            for t_th, r_th in zip(t_thresholds_combined, r_thresholds_combined):
                success_rate = np.mean((t_arr <= t_th) & (r_arr <= r_th)) * 100
                success_rates.append(success_rate)
            
            ax8.bar(x_pos + i*width, success_rates, width, label=f'{split.capitalize()}', 
                   color=colors[split], alpha=0.7)
    
    ax8.set_xlabel('Threshold Pairs (Translation, Rotation)')
    ax8.set_ylabel('Success Rate (%)')
    ax8.set_title('Combined Success Rate')
    ax8.set_xticks(x_pos + width/2)
    ax8.set_xticklabels([f'{t}m, {r}°' for t, r in zip(t_thresholds_combined, r_thresholds_combined)], rotation=45)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Summary statistics table
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    
    table_data = []
    headers = ['Dataset', 'Images', 'Trans Mean (m)', 'Trans Median (m)', 'Rot Mean (°)', 'Rot Median (°)', 
               'Success@(0.25m,5°)', 'Success@(0.5m,10°)', 'Success@(1m,15°)']
    
    for split, (t_err, r_err) in split_errors.items():
        if len(t_err) > 0 and len(r_err) > 0:
            t_arr = np.array(t_err)
            r_arr = np.array(r_err)
            
            row = [
                split.capitalize(),
                len(t_err),
                f'{np.mean(t_err):.3f}',
                f'{np.median(t_err):.3f}',
                f'{np.mean(r_err):.2f}',
                f'{np.median(r_err):.2f}',
                f'{np.mean((t_arr <= 0.25) & (r_arr <= 5.0)) * 100:.1f}%',
                f'{np.mean((t_arr <= 0.5) & (r_arr <= 10.0)) * 100:.1f}%',
                f'{np.mean((t_arr <= 1.0) & (r_arr <= 15.0)) * 100:.1f}%'
            ]
            table_data.append(row)
    
    if table_data:
        table = ax9.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color the header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color the rows based on dataset
        for i, split in enumerate(['train', 'test']):
            if i < len(table_data):
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor(colors[split])
                    table[(i+1, j)].set_alpha(0.3)
    
    plt.suptitle(f'Comprehensive Localization Analysis - {scene_name.upper()}', fontsize=16, fontweight='bold')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined analysis plot saved to: {output_file}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters with comprehensive plotting")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ransac_iters", default=20000, type=int)
    # parser.add_argument("--mlp_dim", default=256, type=int)
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
