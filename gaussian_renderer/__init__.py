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

import math
from typing import Dict

import torch
import torch.nn.functional as F
from gsplat import rasterization

from scene.gaussian.gaussian_model import GaussianModel
from utils.graphics_utils import fov2focal


def get_render_visible_mask(
    pc: GaussianModel, viewpoint_camera, width, height, **rasterize_args
):
    scales = pc.get_scaling

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    rotations = pc.get_rotation
    colors = pc.get_features  # [N, K, 3]
    sh_degree = pc.active_sh_degree
    
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).cuda()
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = width / (2 * tanfovx)
    focal_length_y = height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, width / 2.0],
            [0, focal_length_y, height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        width=width,
        height=height,
        packed=False,
        sh_degree=sh_degree,
        **rasterize_args
    )

    colors.sum().backward()
    render_visible_mask = means3D.grad.norm(dim=-1) > 0
    means3D.grad.zero_()

    return render_visible_mask


def render_gsplat(
    viewpoint_camera,
    pc: GaussianModel,
    bg_color: torch.Tensor,
    override_color=None,
    rgb_only=False,
    features_only=False,
    norm_feat_bf_render=True,
    near_plane=0.01,
    far_plane=10000,
    longest_edge=640,
    **rasterize_args
):
    """
    Render the 3DGS scene.
    Background tensor (bg_color) must be on GPU!
    
    Args:
        features_only: If True, only render feature and score maps, skip RGB rendering
    """
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).cuda() # [4, 4]
    
    if override_color is not None:
        colors = override_color  # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features  # [N, K, 3]
        sh_degree = pc.active_sh_degree

    if bg_color is None:
        bg_color = torch.zeros(3, device="cuda")

    # calculate intrinsic matrix
    width, height = viewpoint_camera.image_width, viewpoint_camera.image_height
    max_edge = max(width, height)
    if max_edge > longest_edge:
        factor = longest_edge / max_edge
        width, height = int(width * factor), int(height * factor)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = width / (2 * tanfovx)
    focal_length_y = height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, width / 2.0],
            [0, focal_length_y, height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    # Initialize return values
    color = None
    radii = None
    visible_mask = None
    render_alphas = None
    info = None

    # render color (skip if features_only is True)
    if not features_only:
        render_colors, render_alphas, info = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=colors,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=bg_color[None],
            width=width,
            height=height,
            packed=False,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            **rasterize_args
        )
        # [1, H, W, 3] -> [3, H, W]
        rendered_image = render_colors[0].permute(2, 0, 1)
        color = rendered_image
        radii = info["radii"].squeeze(0)  # [N, 2]
        radii = torch.square(radii[:, 0]) + torch.square(radii[:, 1]) # [N,]
        visible_mask = radii > 0

        try:
            info["means2d"].retain_grad()  # [1, N, 2]
        except:
            pass
    else:
        # In features_only mode, radii and visible_mask will be set in feature rendering
        pass
    # render feature and score maps
    if rgb_only is False:
        # In features_only mode, use feature rendering for visibility
        if features_only:
            # Use feature rendering to determine visibility and viewspace points
            loc_feature = pc.get_semantic_feature.squeeze()
            if norm_feat_bf_render:
                loc_feature = F.normalize(loc_feature, p=2, dim=-1)

            feat_map, render_alphas, info = rasterization(
                means=means3D,  # [N, 3]
                quats=rotations,  # [N, 4]
                scales=scales,  # [N, 3]
                opacities=opacity.squeeze(-1),  # [N,]
                colors=loc_feature,
                viewmats=viewmat[None],  # [1, 4, 4]
                Ks=K[None],  # [1, 3, 3]
                width=width,
                height=height,
                packed=False,
                near_plane=near_plane,
                far_plane=far_plane,
                **rasterize_args
            )
            feat_map = feat_map[0].permute(2, 0, 1)
            feat_map = F.normalize(feat_map, p=2, dim=0)
            
            # Get visibility info from feature rendering
            radii = info["radii"].squeeze(0)  # [N, 2]
            radii = torch.square(radii[:, 0]) + torch.square(radii[:, 1]) # [N,]
            visible_mask = radii > 0
            
            # Create dummy RGB output
            color = torch.zeros(3, height, width, device="cuda")
            
            # Render score map using the same visible gaussians
            score_feature = pc.get_score_feature[visible_mask].squeeze()
            score_map, alphas, meta = rasterization(
                means=means3D[visible_mask],  # [N, 3]
                quats=rotations[visible_mask],  # [N, 4]
                scales=scales[visible_mask],  # [N, 3]
                opacities=opacity.squeeze(-1)[visible_mask],  # [N,]
                colors=score_feature.unsqueeze(-1),
                viewmats=viewmat[None],  # [1, 4, 4]
                Ks=K[None],  # [1, 3, 3]
                width=width,
                height=height,
                packed=False,
                near_plane=near_plane,
                far_plane=far_plane,
                **rasterize_args
            )
            score_map = score_map[0].permute(2, 0, 1)
            
            try:
                info["means2d"].retain_grad()  # [1, N, 2]
            except:
                pass
        else:
            # Standard mode: render features using visibility from RGB rendering
            loc_feature = pc.get_semantic_feature[visible_mask].squeeze()
            score_feature = pc.get_score_feature[visible_mask].squeeze()
            if norm_feat_bf_render:
                loc_feature = F.normalize(loc_feature, p=2, dim=-1)

            feat_map, alphas, meta = rasterization(
                means=means3D[visible_mask],  # [N, 3]
                quats=rotations[visible_mask],  # [N, 4]
                scales=scales[visible_mask],  # [N, 3]
                opacities=opacity.squeeze(-1)[visible_mask],  # [N,]
                colors=loc_feature,
                viewmats=viewmat[None],  # [1, 4, 4]
                Ks=K[None],  # [1, 3, 3]
                width=width,
                height=height,
                packed=False,
                near_plane=near_plane,
                far_plane=far_plane,
                **rasterize_args
            )
            feat_map = feat_map[0].permute(2, 0, 1)
            feat_map = F.normalize(feat_map, p=2, dim=0)

            score_map, alphas, meta = rasterization(
                means=means3D[visible_mask],  # [N, 3]
                quats=rotations[visible_mask],  # [N, 4]
                scales=scales[visible_mask],  # [N, 3]
                opacities=opacity.squeeze(-1)[visible_mask],  # [N,]
                colors=score_feature.unsqueeze(-1),
                viewmats=viewmat[None],  # [1, 4, 4]
                Ks=K[None],  # [1, 3, 3]
                width=width,
                height=height,
                packed=False,
                near_plane=near_plane,
                far_plane=far_plane,
                **rasterize_args
            )
            score_map = score_map[0].permute(2, 0, 1)

            try: 
                meta["means2d"].retain_grad()  # [1, N, 2]
            except:
                pass
    else:
        feat_map = None
        score_map = None
        
        # If features_only but rgb_only is True, we need to initialize missing values
        if features_only:
            color = torch.zeros(3, height, width, device="cuda")
            radii = torch.zeros(means3D.shape[0], device="cuda")
            visible_mask = torch.zeros(means3D.shape[0], dtype=torch.bool, device="cuda")
            render_alphas = torch.zeros(1, height, width, 1, device="cuda")
            # Create minimal info dict
            info = {"means2d": torch.zeros(1, means3D.shape[0], 2, device="cuda")}

    return {
        "render": color,
        "score_map": score_map,
        "feature_map": feat_map,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "alphas": render_alphas,
    }


def render_from_pose_gsplat(
    pc: GaussianModel,
    pose,
    fovx,
    fovy,
    width,
    height,
    bg_color=None,
    render_mode="RGB+ED",
    rgb_only=False,
    features_only=False,
    norm_feat_bf_render=True,
    near_plane=0.01,
    far_plane=10000,
    **rasterize_args
):
    """
    Render the 3DGS scene from pose.
    """
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    colors = pc.get_features  # [N, K, 3]
    sh_degree = pc.active_sh_degree

    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    focal_length_x = width / (2 * tanfovx)
    focal_length_y = height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, width / 2.0],
            [0, focal_length_y, height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    if bg_color is None:
        bg_color = torch.zeros(3, device="cuda")

    # Initialize return values
    color = None
    depth = None
    radii = None
    visible_mask = None
    render_alphas = None
    info = None

    # render color (skip if features_only is True)
    if not features_only:
        render_colors, render_alphas, info = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=colors,
            viewmats=pose[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=bg_color[None],
            width=int(width),
            height=int(height),
            packed=False,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode=render_mode,
            **rasterize_args
        )
        # [1, H, W, 3] -> [3, H, W]
        rendered_image = render_colors[0].permute(2, 0, 1)
        color = rendered_image[:3]
        if rendered_image.shape[0] == 4:
            depth = rendered_image[3:]
        else:
            depth = None
        radii = info["radii"].squeeze(0)  # [N, 2]
        radii = torch.square(radii[:, 0]) + torch.square(radii[:, 1]) # [N,]
        visible_mask = radii > 0

        try:
            info["means2d"].retain_grad()  # [1, N, 2]
        except:
            pass
    else:
        # For features_only mode, we need to do a minimal render to get visibility info
        dummy_colors = torch.zeros(means3D.shape[0], 1, device=means3D.device)
        dummy_bg = torch.zeros(1, device="cuda")  # Single channel background for dummy colors
        render_colors, render_alphas, info = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=dummy_colors,
            viewmats=pose[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=dummy_bg[None],  # [1, 1] to match single channel
            width=int(width),
            height=int(height),
            packed=False,
            near_plane=near_plane,
            far_plane=far_plane,
            **rasterize_args
        )
        # Create dummy outputs
        color = torch.zeros(3, int(height), int(width), device="cuda")
        depth = None
        radii = info["radii"].squeeze(0)  # [N, 2]
        radii = torch.square(radii[:, 0]) + torch.square(radii[:, 1]) # [N,]
        visible_mask = radii > 0

        try:
            info["means2d"].retain_grad()  # [1, N, 2]
        except:
            pass

    if rgb_only is False:
        loc_feature = pc.get_semantic_feature[visible_mask].squeeze(1)
        score_feature = pc.get_score_feature[visible_mask].squeeze(1)
        if norm_feat_bf_render:
            loc_feature = F.normalize(loc_feature, p=2, dim=-1)

        feat_map, alphas, meta = rasterization(
            means3D[visible_mask],
            rotations[visible_mask],
            scales[visible_mask],
            opacity.squeeze(-1)[visible_mask],
            loc_feature,
            pose[None],
            K[None],
            int(width),
            int(height),
            packed=False,
            near_plane=near_plane,
            far_plane=far_plane,
            **rasterize_args
        )
        feat_map = feat_map[0].permute(2, 0, 1)
        feat_map = F.normalize(feat_map, p=2, dim=0)

        score_map, alphas, meta = rasterization(
            means3D[visible_mask],
            rotations[visible_mask],
            scales[visible_mask],
            opacity.squeeze(-1)[visible_mask],
            score_feature,
            pose[None],
            K[None],
            int(width),
            int(height),
            packed=False,
            near_plane=near_plane,
            far_plane=far_plane,
            **rasterize_args
        )
        score_map = score_map[0].permute(2, 0, 1)

        try:
            meta["means2d"].retain_grad()  # [1, N, 2]
        except:
            pass
    else:
        feat_map = None
        score_map = None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    viewspace_points = info["means2d"]
    if not rgb_only and not features_only:
        # Use meta from feature rendering if available
        try:
            viewspace_points = meta["means2d"]
        except:
            pass
    
    return {
        "render": color,
        "score_map": score_map,
        "feature_map": feat_map,
        "viewspace_points": viewspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "alphas": render_alphas,
        "depth": depth,
    }


# GSplat Driver Methods for Training
def create_gsplat_params(gaussians: GaussianModel) -> Dict[str, torch.nn.Parameter]:
    """Convert GaussianModel to gsplat parameter format"""
    params = {
        "means": torch.nn.Parameter(gaussians.get_xyz.clone()),
        "scales": torch.nn.Parameter(gaussians.get_scaling.clone()),
        "quats": torch.nn.Parameter(gaussians.get_rotation.clone()),
        "opacities": torch.nn.Parameter(gaussians.get_opacity.squeeze(-1).clone()),
        "sh0": torch.nn.Parameter(gaussians._features_dc.clone()),
        "shN": torch.nn.Parameter(gaussians._features_rest.clone()),
        "loc_feature": torch.nn.Parameter(gaussians._semantic_feature.clone()),
        "score_feature": torch.nn.Parameter(gaussians._score_feature.clone()),
    }
    return params


def create_gsplat_optimizers(params: Dict[str, torch.nn.Parameter], opt_param) -> Dict[str, torch.optim.Optimizer]:
    """Create optimizers for gsplat parameters following original learning rates"""
    lr_init = opt_param.position_lr_init
    lr_final = opt_param.position_lr_final
    lr_delay_mult = opt_param.position_lr_delay_mult
    lr_max_steps = opt_param.position_lr_max_steps
    
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=lr_init),
        "scales": torch.optim.Adam([params["scales"]], lr=opt_param.scaling_lr),
        "quats": torch.optim.Adam([params["quats"]], lr=opt_param.rotation_lr),
        "opacities": torch.optim.Adam([params["opacities"]], lr=opt_param.opacity_lr),
        "sh0": torch.optim.Adam([params["sh0"]], lr=opt_param.feature_lr),
        "shN": torch.optim.Adam([params["shN"]], lr=opt_param.feature_lr / 20.0),
        "loc_feature": torch.optim.Adam([params["loc_feature"]], lr=opt_param.feature_lr),
        "score_feature": torch.optim.Adam([params["score_feature"]], lr=opt_param.feature_lr),
    }
    return optimizers


def update_learning_rates(optimizers: Dict[str, torch.optim.Optimizer], iteration: int, opt_param):
    """Update learning rates for position parameters with exponential decay"""
    lr_init = opt_param.position_lr_init
    lr_final = opt_param.position_lr_final
    lr_delay_mult = opt_param.position_lr_delay_mult
    lr_max_steps = opt_param.position_lr_max_steps
    
    # Exponential decay for position learning rate
    if lr_max_steps:
        lr = lr_final + (lr_init - lr_final) * max(0, 1 - iteration / lr_max_steps) ** lr_delay_mult
    else:
        lr = lr_init
    
    for param_group in optimizers["means"].param_groups:
        param_group['lr'] = lr


def render_with_gsplat(viewpoint_cam, params: Dict[str, torch.nn.Parameter], background, 
                      rgb_only=False, features_only=False, norm_feat_bf_render=True, **kwargs):
    """Render using gsplat rasterization"""
    # Prepare camera parameters
    width, height = viewpoint_cam.image_width, viewpoint_cam.image_height
    viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()
    
    # Calculate intrinsic matrix
    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)
    focal_length_x = width / (2 * tanfovx)
    focal_length_y = height / (2 * tanfovy)
    K = torch.tensor([
        [focal_length_x, 0, width / 2.0],
        [0, focal_length_y, height / 2.0],
        [0, 0, 1],
    ], device="cuda")
    
    # Prepare colors (combine sh0 and shN for spherical harmonics)
    sh_degree = 3  # Assuming max SH degree is 3
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    
    # Initialize return values
    rendered_image = None
    render_alphas = None
    info = None
    
    # Render RGB (skip if features_only is True)
    if not features_only:
        render_colors, render_alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"],
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            backgrounds=background[None] if background is not None else None,
            width=width,
            height=height,
            packed=False,
            sh_degree=sh_degree,
            **kwargs
        )
        # Convert output format: [1, H, W, 3] -> [3, H, W]
        rendered_image = render_colors[0].permute(2, 0, 1)
    else:
        # For features_only mode, we need to do a minimal render to get visibility info
        dummy_colors = torch.zeros(params["means"].shape[0], 1, device=params["means"].device)
        dummy_bg = torch.zeros(1, device="cuda")  # Single channel background for dummy colors
        render_colors, render_alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"],
            colors=dummy_colors,
            viewmats=viewmat[None],
            Ks=K[None],
            backgrounds=dummy_bg[None],  # [1, 1] to match single channel
            width=width,
            height=height,
            packed=False,
            **kwargs
        )
        # Create dummy RGB image
        rendered_image = torch.zeros(3, height, width, device="cuda")
    
    # Calculate visibility filter
    radii = info["radii"].squeeze(0)
    radii_norm = torch.square(radii[:, 0]) + torch.square(radii[:, 1])
    visibility_filter = radii_norm > 0
    
    result = {
        "render": rendered_image,
        "viewspace_points": info["means2d"],
        "visibility_filter": visibility_filter,
        "radii": radii_norm,
        "alphas": render_alphas,
    }
    
    if not rgb_only:
        # Render feature map
        visible_indices = visibility_filter
        if visible_indices.sum() > 0:
            loc_feature = params["loc_feature"][visible_indices].squeeze(1)
            score_feature = params["score_feature"][visible_indices].squeeze(1)
            
            if norm_feat_bf_render:
                loc_feature = F.normalize(loc_feature, p=2, dim=-1)
            
            # Render feature map
            feat_colors, _, _ = rasterization(
                means=params["means"][visible_indices],
                quats=params["quats"][visible_indices],
                scales=params["scales"][visible_indices],
                opacities=params["opacities"][visible_indices],
                colors=loc_feature,
                viewmats=viewmat[None],
                Ks=K[None],
                width=width,
                height=height,
                packed=False,
                **kwargs
            )
            feat_map = feat_colors[0].permute(2, 0, 1)
            feat_map = F.normalize(feat_map, p=2, dim=0)
            
            # Render score map
            score_colors, _, _ = rasterization(
                means=params["means"][visible_indices],
                quats=params["quats"][visible_indices], 
                scales=params["scales"][visible_indices],
                opacities=params["opacities"][visible_indices],
                colors=score_feature,
                viewmats=viewmat[None],
                Ks=K[None],
                width=width,
                height=height,
                packed=False,
                **kwargs
            )
            score_map = score_colors[0].permute(2, 0, 1)
            
            result["feature_map"] = feat_map
            result["score_map"] = score_map
        else:
            result["feature_map"] = None
            result["score_map"] = None
    
    return result


def sync_gaussians_from_params(gaussians: GaussianModel, params: Dict[str, torch.nn.Parameter]):
    """Sync gsplat parameters back to GaussianModel for saving/checkpointing"""
    gaussians._xyz.data = params["means"].data
    gaussians._scaling.data = torch.log(params["scales"].data)
    gaussians._rotation.data = params["quats"].data
    gaussians._opacity.data = torch.logit(params["opacities"].data.unsqueeze(-1), eps=1e-6)
    gaussians._features_dc.data = params["sh0"].data
    gaussians._features_rest.data = params["shN"].data
    gaussians._semantic_feature.data = params["loc_feature"].data
    gaussians._score_feature.data = params["score_feature"].data
