import os
import torch
import open3d as o3d
import cv2
import numpy as np
import torchvision
from PIL import Image
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import Camera
from plyfile import PlyData
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, get_args
from encoders.superpoint.lightglue import LightGlue
from encoders.superpoint.superpoint import SuperPoint
from utils.match_img import score_feature_one
from eval import save_matchimg, backproject_depth, umeyama_alignment, compute_registration_error_w_scale
from utils.metrics_match import compute_metrics
from pprint import pprint
from utils.vis_scoremap import one_channel_vis



SP_THRESHOLD = 0.01
LG_THRESHOLD = 0.01

matcher = LightGlue({
            "filter_threshold": LG_THRESHOLD#0.01,
        }).to("cuda").eval()

conf = {
    "sparse_outputs": True,
    "dense_outputs": True,
    "max_num_keypoints": 1024,
    "detection_threshold": SP_THRESHOLD #0.01,
}
encoder = SuperPoint(conf).to("cuda").eval()


scene_name = "scene0765_00"
test_fold_path=f"/home/koki/code/cc/feature_3dgs_2/gsreg/scannet_test_final/{scene_name}"
os.makedirs(test_fold_path, exist_ok=True)
scene_path = f"/home/koki/code/cc/feature_3dgs_2/gsreg/test/{scene_name}"



def project_points(points, intrinsic, extrinsic, ref_trans):
    """
    Projects 3D points onto the 2D image plane using camera intrinsic and extrinsic matrices.

    Parameters:
    - points: Nx3 numpy array of 3D points.
    - intrinsic: 3x3 numpy array representing the camera intrinsic matrix.
    - extrinsic: 4x4 numpy array representing the camera extrinsic matrix.

    Returns:
    - projected_points: Nx2 numpy array of 2D points in the image plane.
    - depths: N-element array of depths for each point.
    """
    # Convert points to homogeneous coordinates (Nx4)
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    # extrinsic = ref_trans@extrinsic
    ref_inv = np.linalg.inv(ref_trans)

    # Apply extrinsic transformation
    points_camera = (extrinsic @ ref_inv @ points_hom.T).T

    # Depths are in the z-axis after the extrinsic transform
    depths = points_camera[:, 2]

    # Filter points that are in front of the camera
    valid_idx = depths > 0
    points_camera = points_camera[valid_idx]
    depths = depths[valid_idx]

    # Apply intrinsic transformation to get image coordinates
    projected_points = (intrinsic @ points_camera[:, :3].T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2, None]
    # breakpoint()
    return projected_points, depths, valid_idx

def count_points_in_view(points, intrinsic, extrinsic, ref_list, width, height):
    """
    Counts how many 3D points fall within the image bounds of a camera.

    Parameters:
    - points: Nx3 numpy array of 3D points.
    - intrinsic: 3x3 numpy array representing the camera intrinsic matrix.
    - extrinsic: 4x4 numpy array representing the camera extrinsic matrix.
    - width: Width of the image plane in pixels.
    - height: Height of the image plane in pixels.

    Returns:
    - count: Number of points within the camera's field of view.
    """
    projected_points, depths, valid_idx = project_points(points, intrinsic, extrinsic, ref_list)

    # Check if projected points are within image boundaries
    in_view = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) & \
              (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)

    # Return the count of points that are within the view
    return np.sum(in_view)



def main():
    overlap_path = f"{scene_path}/ref_overlap_largest.ply"
    # overlap_path = f"{scene_path}/A/outputs/0/point_cloud/iteration_10000/point_cloud.ply"

    overlap_ply = PlyData.read(overlap_path)

    x = np.asarray(overlap_ply.elements[0].data['x'])
    y = np.asarray(overlap_ply.elements[0].data['y'])
    z = np.asarray(overlap_ply.elements[0].data['z'])
    points_overlap = np.stack([x,y,z], axis=1)




    estimated = np.load(f"{scene_path}/estimated_transform_scale.npy")
    gt = np.load(f"{scene_path}/gt_transformation.npy")



    parser_ref = ArgumentParser(description="Testing script parameters")
    model_ref = ModelParams(parser_ref, sentinel=True)
    pipe_ref = PipelineParams(parser_ref)
    model_ref._model_path = f"{scene_path}/A/outputs/0"
    args_ref = get_args(model_ref)
    print(args_ref.model_path)
    gaussians_ref = GaussianModel(args_ref.sh_degree)
    scene_ref = Scene(args_ref, gaussians_ref, load_iteration=10000, shuffle=False)




    parser_src = ArgumentParser(description="Testing script parameters")
    model_src = ModelParams(parser_src, sentinel=True)
    pipe_src = PipelineParams(parser_src)
    model_src._model_path = f"{scene_path}/B/outputs/0"
    args_src = get_args(model_src)
    print(args_src.source_path)
    gaussians_src = GaussianModel(args_src.sh_degree)
    scene_src = Scene(args_src, gaussians_src, load_iteration=10000, shuffle=False)




    scannet_path = "/home/koki/code/cc/feature_3dgs_2/gsreg"
    scene_list = np.load(os.path.join(scannet_path, f'test_transformations.npz'), allow_pickle=True)['transformations'].item()

    gt_transform_list = scene_list['gt_transformations_list']
    ref_transform_list = scene_list['ref_transformations_list']
    src_transform_list = scene_list['src_transformations_list']


    ref_list = ref_transform_list[scene_name]




    image_width=1296
    image_height=968


    # Track the best camera
    max_in_view_count = 0
    best_c_index = -1

    # Evaluate each camera
    for i, cam in enumerate(scene_ref.train_cameras[1.0]):
        in_view_count = count_points_in_view(points_overlap, cam.intrinsic_matrix, cam.extrinsic_matrix, ref_list, 
                                            image_width, image_height)
        print(i, in_view_count)
        if in_view_count > max_in_view_count:
            max_in_view_count = in_view_count
            best_c_index = i

    print('best camera: ',best_c_index, max_in_view_count)



    view_ref = scene_ref.train_cameras[1.0][best_c_index]

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_pkg_ref = render(view_ref, gaussians_ref, pipe_ref, bg_color=background)
    torchvision.utils.save_image(render_pkg_ref["render"], f"{test_fold_path}/ref.png")

    feature_ref = render_pkg_ref["feature_map"][:16]
    feature_ref = feature_ref.detach().cpu().numpy().astype(np.float16)
    torch.save(torch.tensor(feature_ref).half(), f"{test_fold_path}/ref_fmap.pt")

    score_ref = render_pkg_ref['score_map']
    score_ref = score_ref.detach().cpu().numpy().astype(np.float16)
    torch.save(torch.tensor(score_ref).half(), f"{test_fold_path}/ref_smap.pt")

    depth_ref = render_pkg_ref['depth']
    depth_ref = depth_ref.detach().cpu().numpy().astype(np.float16)
    torch.save(torch.tensor(depth_ref).half(), f"{test_fold_path}/ref_depth.pt")

    torch.save(torch.tensor(view_ref.extrinsic_matrix), f"{test_fold_path}/ref_ex.pt")
    torch.save(torch.tensor(view_ref.intrinsic_matrix), f"{test_fold_path}/ref_in.pt")




    src_trans = np.load(f"{scene_path}/estimated_inv_trans.npy")
    print(src_trans)
    E_new = src_trans @ view_ref.extrinsic_matrix

    print(view_ref.extrinsic_matrix)
    print(E_new)


    view_src = scene_src.train_cameras[1.0][0]
    view_src = Camera(colmap_id=view_src.colmap_id, R=np.transpose(E_new[:3, :3]), 
                    T=E_new[:3, 3], 
                    FoVx=view_src.FoVx, FoVy=view_src.FoVy,
                    image=view_src.original_image, gt_alpha_mask=None, image_name=view_src.image_name, uid=view_ref.uid,
                    semantic_feature=view_src.semantic_feature, score_feature=view_src.score_feature, 
                    intrinsic_params=view_src.intrinsic_params)

    render_pkg_src = render(view_src, gaussians_src, pipe_src, bg_color=background)
    torchvision.utils.save_image(render_pkg_src["render"], f"{test_fold_path}/src.png")

    feature_src = render_pkg_src["feature_map"][:16]
    feature_src = feature_src.detach().cpu().numpy().astype(np.float16)
    torch.save(torch.tensor(feature_src).half(), f"{test_fold_path}/src_fmap.pt")

    score_src = render_pkg_src['score_map']
    score_src = score_src.detach().cpu().numpy().astype(np.float16)
    torch.save(torch.tensor(score_src).half(), f"{test_fold_path}/src_smap.pt")

    depth_src = render_pkg_src['depth']
    depth_src = depth_src.detach().cpu().numpy().astype(np.float16)
    torch.save(torch.tensor(depth_src).half(), f"{test_fold_path}/src_depth.pt")

    torch.save(torch.tensor(view_src.extrinsic_matrix), f"{test_fold_path}/src_ex.pt")
    torch.save(torch.tensor(view_src.intrinsic_matrix), f"{test_fold_path}/src_in.pt")






def match_img():
    def load():

        img0 = np.array(Image.open(f"{test_fold_path}/ref.png"))
        img1 = np.array(Image.open(f"{test_fold_path}/src.png"))
        sp0 = f"{test_fold_path}/ref_smap.pt"
        sp1 = f"{test_fold_path}/src_smap.pt"
        s0 = torch.load(sp0)
        s1 = torch.load(sp1)

        fp0 = f"{test_fold_path}/ref_fmap.pt"
        fp1 = f"{test_fold_path}/src_fmap.pt"
        f0 = torch.load(fp0)
        f1 = torch.load(fp1)

        K_0 = torch.load(f"{test_fold_path}/ref_in.pt").float().detach().clone().cpu()
        K_1 = torch.load(f"{test_fold_path}/src_in.pt").float().detach().clone().cpu()


        T0 = torch.load(f"{test_fold_path}/ref_ex.pt").float().detach().clone().cpu()
        T1 = torch.load(f"{test_fold_path}/src_ex.pt").float().detach().clone().cpu()
        # epsilon = 1e-8 
        # T0 = T0 + np.eye(T0.shape[0]) * epsilon

        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)
        T_1to0 = T_0to1.inverse()

        data = {
            "img0": img0,
            "img1": img1,
            # "img_orig0": img_orig0,
            # "img_orig1": img_orig1,
            "s0": s0,
            "s1": s1,
            "ft0": f0,
            "ft1": f1,
            "K0": K_0,
            "K1": K_1,
            "T_0to1": T_0to1,
            "T_1to0": T_1to0,
        }
        return data
    
    data_fm = load()
    score_feature_one(data_fm, matcher, threshold=0.1)
    save_matchimg(data_fm, f"{test_fold_path}/match.png")
    compute_metrics(data_fm)


    pprint(data_fm['inliers'][0].sum()/len(data_fm['inliers'][0]))
    pprint(data_fm['R_errs'])
    pprint(data_fm['t_errs'])
    pprint(data_fm['epi_errs'][0].max())

    torch.save(data_fm['mkpt0'], f"{test_fold_path}/ref_mkpt.pt")
    torch.save(data_fm["mkpt1"], f"{test_fold_path}/src_mkpt.pt")

    # score_map_vis = one_channel_vis(score_map)
    # score_map_vis.save(os.path.join(score_map_path, '{0:05d}'.format(idx) + "_score_vis.png"))

    breakpoint()




def find_final_trans():
    estimated = np.load(f"{scene_path}/estimated_transform_scale.npy")
    gt = np.load(f"{scene_path}/gt_transformation.npy")
    src_trans = np.load(f"{scene_path}/estimated_inv_trans.npy")

    mkpt0 = torch.load(f"{test_fold_path}/ref_mkpt.pt")
    mkpt1 = torch.load(f"{test_fold_path}/src_mkpt.pt")
    d0 = torch.load(f"{test_fold_path}/ref_depth.pt")
    d1 = torch.load(f"{test_fold_path}/src_depth.pt")

    intrin0 = torch.load(f"{test_fold_path}/ref_in.pt")
    intrin1 = torch.load(f"{test_fold_path}/src_in.pt")
    ex0 = torch.load(f"{test_fold_path}/ref_ex.pt")
    ex1 = torch.load(f"{test_fold_path}/src_ex.pt")

    pixel_thr = 0.5
    ransac_thr = pixel_thr / np.mean([intrin0[0, 0], intrin1[1, 1], intrin0[0, 0], intrin1[1, 1]])
    E, mask = cv2.findEssentialMat(
        mkpt0.numpy(), mkpt1.numpy(), np.eye(3), threshold=ransac_thr, prob=0.99999, method=cv2.RANSAC)
    mask = mask.flatten()
    # mkpt0 = mkpt0[mask == 1]
    # mkpt1 = mkpt1[mask == 1]

    # breakpoint()

    p3d0 = backproject_depth(d0, mkpt0, intrin0, ex0)
    p3d1 = backproject_depth(d1, mkpt1, intrin1, ex1)

    

    T = umeyama_alignment(source_points=p3d1, target_points=p3d0, with_scaling=True)
    T = T.numpy()
    T_inv = np.linalg.inv(T)
    
    print(T)
    print(T_inv)
    print()
    print(src_trans)
    print()
    print(T@src_trans)
    print(src_trans@T)
    print()
    print(T_inv@src_trans)
    print(src_trans@T_inv)
    rre, rte, rse = compute_registration_error_w_scale(gt, estimated)



    breakpoint()



    # d0_vis = one_channel_vis(d0)
    # d0_vis.save(f"{test_fold_path}/ref_depth_vis.png")

    # d1_vis = one_channel_vis(d1)
    # d1_vis.save(f"{test_fold_path}/src_depth_vis.png")




# python registration.py
if __name__=="__main__":
    match_img()



# # colors = np.array(colors)
# points_ref_trans = (ref_list @ points_homogeneous.T).T
# points_ref_trans = points_ref_trans[:, :3]


# pcd = o3d.geometry.PointCloud()
# # Assign points to the PointCloud

# pcd.points = o3d.utility.Vector3dVector(points_ref_trans)
# # pcd.colors = o3d.utility.Vector3dVector(colors)
# pcd.paint_uniform_color([0, 1, 0])
# o3d.io.write_point_cloud("cameras.ply", pcd)





