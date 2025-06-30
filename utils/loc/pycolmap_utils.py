import numpy as np
import pycolmap

np.random.seed(1000)

def opencv_to_pycolmap_pnp(db_world, q_matched, K, image_width, image_height):
    """
    Convert OpenCV PnP inputs to pycolmap format and estimate pose
    
    Args:
        db_world: Nx3 array of 3D world points
        q_matched: Nx2 array of 2D image points
        K: 3x3 camera intrinsic matrix
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
    
    Returns:
        dict containing pose estimation results
    """
    points3D = np.asarray(db_world, dtype=np.float64)
    points2D = np.asarray(q_matched, dtype=np.float64)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    camera = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=image_width,
        height=image_height,
        params=[fx, cx, cy]
    )
    estimation_options = pycolmap.AbsolutePoseEstimationOptions()
    estimation_options.ransac.max_error = 4.0  # Stricter threshold, typically 2-4 pixels
    estimation_options.ransac.min_inlier_ratio = 0.2  # Require more inliers
    estimation_options.ransac.confidence = 0.9999
    estimation_options.ransac.min_num_trials = 1000  # Ensure enough RANSAC iterations
    estimation_options.ransac.max_num_trials = 10000
    # Set up refinement options
    refinement_options = pycolmap.AbsolutePoseRefinementOptions()
    # Estimate pose
    result = pycolmap.estimate_and_refine_absolute_pose(
        points2D=points2D,
        points3D=points3D,
        camera=camera,
        estimation_options=estimation_options,
        refinement_options=refinement_options
    )
    R, t = convert_pycolmap_pose(result)
    return R, t


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix.
    Args:
        qvec (ndarray): Quaternion in format [x, y, z, w]
    Returns:
        ndarray: 3x3 rotation matrix
    """
    # Normalize quaternion
    qvec = qvec / np.linalg.norm(qvec)
    x, y, z, w = qvec
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y + 2*w*z, 2*x*z - 2*w*y],
        [2*x*y - 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z + 2*w*x],
        [2*x*z + 2*w*y, 2*y*z - 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

def convert_pycolmap_pose(result):
    """
    Convert pycolmap pose result to rotation matrix and translation vector.
    The pose remains in world-to-camera coordinates.
    
    Args:
        result (dict): Result from pycolmap.estimate_and_refine_absolute_pose
        
    Returns:
        R (ndarray): 3x3 rotation matrix (world to camera)
        t (ndarray): 3x1 translation vector in meters (world to camera)
    """
    if result is None:
        return None, None
    R = qvec2rotmat(result['cam_from_world'].rotation.quat).T
    t = result['cam_from_world'].translation.reshape(3,1)
    
    return R, t
