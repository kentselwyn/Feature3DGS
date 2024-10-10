import torch
import os
import numpy as np
from scene.dataset_readers import readColmapSceneInfo
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from scene import Scene
from scene.gaussian_model import GaussianModel
from match_images import matchimg2
from PIL import Image
from codes.metrics_match import compute_metrics

def test():
    path = "/home/koki/code/cc/feature_3dgs_2/all_data/scene0755_00/A"

    scene_info = readColmapSceneInfo(path, foundation_model="imrate:2", eval=False)
    cam = scene_info.train_cameras[0]
    print(cam.width)
    print(cam.R)
    print(cam.T)
    print(cam.image_name)


    train_cameras = {}
    intrinsic_path = "/home/koki/code/cc/feature_3dgs_2/all_data/scannet_test_1500_info/intrinsics.npz"
    intrinsics = dict(np.load(intrinsic_path))
    print(intrinsics['scene0755_00'])

    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)




class GroupParams():
    def __init__(self):
        self.sh_degree = 3
        self.source_path = ""
        self.foundation_model = "" ###
        self.model_path = ""
        self.images = None
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.speedup = False ###
        self.render_items = ['RGB', 'Depth', 'Edge', 'Normal', 'Curvature', 'Feature Map', 'Score Map']



# python gt_generation.py
if __name__=="__main__":
    scene_name = "scene0000_01"
    lp = GroupParams()
    lp.source_path = f"/home/koki/code/cc/feature_3dgs_2/all_data/{scene_name}/A"
    lp.foundation_model = "imrate:2_th:0.01_mlpdim:16"
    lp.model_path = f"{lp.source_path}/outputs/imrate:2_th:0.01_mlpdim:16"

    gaussians = GaussianModel(lp.sh_degree)
    scene = Scene(lp, gaussians, shuffle=False)

    train_cams = scene.getTrainCameras()

    num0 = 62
    num1 = 87
    out0 = f"{num0:05d}"
    out1 = f"{num1:05d}"


    cam0 = train_cams[num0]
    cam1 = train_cams[num1]

    Render_type = "trains"

    img0 = Image.open(f"{lp.model_path}/rendering/{Render_type}/ours_7000/image_renders/{out0}.png")
    img1 = Image.open(f"{lp.model_path}/rendering/{Render_type}/ours_7000/image_renders/{out1}.png")
    img0 = np.array(img0)
    img1 = np.array(img1)

    H, W, _ = img0.shape


    sp0 = f"{lp.model_path}/rendering/{Render_type}/ours_7000/score_tensors/{out0}_smap_CxHxW.pt"
    sp1 = f"{lp.model_path}/rendering/{Render_type}/ours_7000/score_tensors/{out1}_smap_CxHxW.pt"
    s0 = torch.load(sp0)
    s1 = torch.load(sp1)

    fp0 = f"{lp.model_path}/rendering/{Render_type}/ours_7000/feature_tensors/{out0}_fmap_CxHxW.pt"
    fp1 = f"{lp.model_path}/rendering/{Render_type}/ours_7000/feature_tensors/{out1}_fmap_CxHxW.pt"
    f0 = torch.load(fp0)
    f1 = torch.load(fp1)


    K_0 = torch.tensor(cam0.intrinsic_matrix, dtype=torch.float)
    K_1 = torch.tensor(cam1.intrinsic_matrix, dtype=torch.float)

    K_0[:2, :] = K_0[:2, :] * 0.5
    K_1[:2, :] = K_1[:2, :] * 0.5


    T0 = cam0.extrinsic_matrix
    T1 = cam1.extrinsic_matrix

    T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)
    T_1to0 = T_0to1.inverse()
    
    
    draw_name = f"{scene_name}_{Render_type}_{num0}_{num1}"
    m_kpts0, m_kpts1 = matchimg2(img0, img1, s0, s1, f0, f1,mlp_dim=16, draw_name=draw_name)
    

    data = {
        'T_0to1': T_0to1,   # (4, 4)
        'T_1to0': T_1to0,
        'K0': K_0,  # (3, 3)
        'K1': K_1,
        'dataset_name': 'ScanNet',
        'scene_id': scene_name,
        "mkpt0": m_kpts0,
        "mkpt1": m_kpts1,
    }

    compute_metrics(data)
    print(data['mkpt0'].shape)
    print(data['R_errs'])
    print(data['t_errs'])
    print(f"{data['inliers'][0].sum()}/{len(data['inliers'][0])}")
    print(data['epi_errs'])


    breakpoint()





