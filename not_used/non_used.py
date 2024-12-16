import torch
import numpy as np





def backproject_depth(depth_map, kpts, intrin, extrin):
    intrin_inv = torch.linalg.inv(intrin).float()
    extrin_inv = torch.linalg.inv(extrin).float()

    d_values = depth_map[0, kpts[:, 1].int(), kpts[:, 0].int()]
    ones = torch.ones((kpts.shape[0], 1))
    kpt_homo = torch.cat([kpts, ones], dim=1).float()
    p_cam_norm = intrin_inv @ kpt_homo.T

    p_cam = (p_cam_norm * d_values).T
    p_cam_homo = torch.cat([p_cam, torch.ones((p_cam.shape[0], 1))], dim=1)
    p_world = (extrin_inv @ p_cam_homo.T).T

    return p_world[:, :3]

    
def umeyama_alignment(source_points, target_points, with_scaling=True):

    assert source_points.shape == target_points.shape, "Source and target must have the same shape."
    N = source_points.shape[0]

    # Step 1: Compute centroids
    mu_X = source_points.mean(dim=0)
    mu_Y = target_points.mean(dim=0)

    X_centered = source_points - mu_X
    Y_centered = target_points - mu_Y

    # Step 3: Compute covariance matrix
    Sigma_XY = (Y_centered.T @ X_centered) / N

    U, S, Vt = torch.linalg.svd(Sigma_XY, full_matrices=True)
    V = Vt.T

    # Step 5: Compute rotation matrix
    D = torch.eye(3)
    if torch.det(U @ V.T) < 0:
        D[-1, -1] = -1
    
    R = U @ D @ V.T

    # Step 6: Compute scaling factor
    if with_scaling:
        var_X = X_centered.pow(2).sum() / N
        s = (S * D.diag()).sum() / var_X
    else:
        s = 1.0

    # Step 7: Compute translation vector
    t = mu_Y - s * R @ mu_X

    # Step 8: Form the transformation matrix
    T = torch.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t

    return T


def compute_registration_error_w_scale(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transform (array): ground truth transformation matrix (4, 4)
        est_transform (array): estimated transformation matrix (4, 4)

    Returns:
        rre (float): relative rotation error.
        rte (float): relative translation error.
    """
    gt_rotation, gt_translation, gt_scale = get_rotation_translation_from_transform_w_scale(gt_transform)
    est_rotation, est_translation, est_scale = get_rotation_translation_from_transform_w_scale(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    rse = compute_relative_scale_error(gt_scale, est_scale)
    return rre, rte, rse

from typing import Tuple
def get_rotation_translation_from_transform_w_scale(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    scale = ((transform[:3, :3] @ transform[:3, :3].T).trace() / 3) ** 0.5
    rotation = transform[:3, :3] / scale
    translation = transform[:3, 3] / scale
    return rotation, translation, scale 

def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre

def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation) / np.linalg.norm(gt_translation)

def compute_relative_scale_error(gt_scale: np.ndarray, est_scale: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_scale - est_scale) / np.linalg.norm(gt_scale)



