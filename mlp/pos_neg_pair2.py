import os
import torch
from matchers.lightglue import LightGlue
from encoders.superpoint.superpoint import SuperPoint
# lightglue match
from PIL import Image
from utils.general_utils import PILtoTorch
import numpy as np
import utils.loc_utils as loc_utils
from pathlib import Path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_args
from scene import Scene, GaussianModel
import torch.nn.functional as F
from utils.metrics_match import compute_metrics
from z_scannet1500.utils.utils import print_eval_to_file, save_matchimg
from copy import deepcopy
from encoders.superpoint.mlp import get_mlp_model, get_mlp_dataset, get_mlp_augment, MLP_module_16_short


def get_pos_pairs():
    SCENE = "scene_stairs"
    SAVE_PATH = f"/home/koki/code/cc/feature_3dgs_2/mlp/imgs/{SCENE}"
    os.makedirs(SAVE_PATH, exist_ok=True)
    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": 1024,
        "detection_threshold": 0.01,
    }
    encoder = SuperPoint(conf).cuda().eval()
    matcher = LightGlue({"filter_threshold": 0.01 ,}).cuda().eval()
    # p0 = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/{SCENE}/train/rgb/seq-02-frame-000320.color.png"
    # p1 = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/{SCENE}/train/rgb/seq-05-frame-000146.color.png"
    # p0 = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/{SCENE}/train/rgb/seq-02-frame-000006.color.png"
    p0 = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/{SCENE}/train/rgb/seq-06-frame-000461.color.png"
    p1 = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/{SCENE}/train/rgb/seq-03-frame-000498.color.png"
    n0 = Path(p0).stem
    n1 = Path(p1).stem

    im0 = np.array(Image.open(p0))
    im1 = np.array(Image.open(p1))
    
    im0 = torch.from_numpy(im0) / 255.0
    im0 = im0.permute(2,0,1).unsqueeze(0).cuda()
    im1 = torch.from_numpy(im1) / 255.0
    im1 = im1.permute(2,0,1).unsqueeze(0).cuda()
    tmp = {}
    pred0 = encoder({"image": im0})
    pred1 = encoder({"image": im1})
    tmp["descriptors0"] = pred0["descriptors"]
    tmp["descriptors1"] = pred1["descriptors"]
    tmp["keypoints0"] = pred0["keypoints"]
    tmp["keypoints1"] = pred1["keypoints"]
    tmp["image_size"] = im0.shape[2:]

    # breakpoint()
    pred = matcher(tmp)
    
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[0][valid]].cpu()
    result = {}
    result['mkpt0'] = m0
    result['mkpt1'] = m1
    result['kpt0'] = tmp["keypoints0"][0].cpu()
    result['kpt1'] = tmp["keypoints1"][0].cpu()
    # tmp["image"] = render_q
    result['img0'] = im0.squeeze(0).permute(1, 2, 0)
    result['img1'] = im1.squeeze(0).permute(1, 2, 0)
    # breakpoint()
    loc_utils.save_matchimg(result, f'{SAVE_PATH}/{n0}_{n1}.png')





def find_match(data, mlp_usage):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": 1024,
        "detection_threshold": 0.01,
    }
    encoder = SuperPoint(conf).cuda().eval()
    pred0 = encoder({"image": data["img0"]})
    pred1 = encoder({"image": data["img1"]})
    tmp = {}
    tmp["descriptors0"] = pred0["descriptors"]
    tmp["descriptors1"] = pred1["descriptors"]
    tmp["keypoints0"] = pred0["keypoints"]
    tmp["keypoints1"] = pred1["keypoints"]
    tmp["image_size"] = data["img0"].shape[2:]
    desc0 = tmp["descriptors0"][0]  # [242, 256]
    desc1 = tmp["descriptors1"][0]  # [235, 256]

    mlp = MLP_module_16_short().to(device)
    CKPT_FOLDER = Path(f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc")
    model_path = CKPT_FOLDER/f"7_scenes/pgt_7scenes_stairs/mlpckpt/type:SP_time:20250107_160254_dim16_batch64_lr0.0008_epoch5000/epoch_1722.pt"
    # model_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train/sparse/mlp2/ckpts/20250521_113506/epoch_212.pt"
    ckpt = torch.load(model_path)
    mlp.load_state_dict(ckpt)

    
    
    desc0 = F.normalize(desc0, dim=1)  # [242, 256]
    desc1 = F.normalize(desc1, dim=1)  # [235, 256]

    if mlp_usage:
        desc0 = mlp.decode(mlp(desc0))
        desc1 = mlp.decode(mlp(desc1))
        SAVE_PATH = f"/home/koki/code/cc/feature_3dgs_2/mlp/imgs/test_mlp_ori"
    else:
        SAVE_PATH = f"/home/koki/code/cc/feature_3dgs_2/mlp/imgs/test"
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    N0, N1 = desc0.shape[0], desc1.shape[0]

    sim = torch.matmul(desc0, desc1.T) 
    nn12 = sim.argmax(dim=1).to(torch.int32)
    nn21 = sim.argmax(dim=0).to(torch.int32)
    m0 = torch.full((N0,), -1, dtype=torch.int32).cuda()
    m1 = torch.full((N1,), -1, dtype=torch.int32).cuda()

    i_idx = torch.arange(N0, dtype=torch.int32).cuda()
    j_idx = nn12
    validd = (nn21[j_idx] == i_idx)
    # breakpoint()
    m0[validd] = j_idx[validd]
    m1[j_idx[validd]] = i_idx[validd]
    # for i in range(N0):
    #     j = nn12[i].item()
    #     if nn21[j].item() == i:
    #         m0[i] = j
    #         m1[j] = i
    
    
    valid = (m0 > -1)
    print(data["identifiers"])
    # mapper0 = torch.arange(m0.shape[0])[valid]
    # mapper1 = torch.arange(m1.shape[0])[m1>-1]
    # mapper1 = torch.arange(m1.shape[0])[valid]

    mat0, mat1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[valid]].cpu()
    # desc0_, desc1_ = tmp["descriptors0"][0].cpu(), tmp["descriptors1"][0].cpu()
    # mat_desc0 = desc0_[valid]
    # mat_desc1 = desc1_[m0[valid]]
    # result = {}
    # result['mkpt0'] = mat0
    # result['mkpt1'] = mat1
    # result['kpt0'] = tmp["keypoints0"][0].cpu()
    # result['kpt1'] = tmp["keypoints1"][0].cpu()
    # result['img0'] = data["img0"].squeeze(0).permute(1, 2, 0).cpu()
    # result['img1'] = data["img1"].squeeze(0).permute(1, 2, 0).cpu()
    # loc_utils.save_matchimg(result, f'{data["identifiers"][0]}.png')
    data["desc0"] = deepcopy(tmp["descriptors0"][0].detach().cpu())
    data["desc1"] = deepcopy(tmp["descriptors1"][0].detach().cpu())
    data['mkpt0'] = mat0.cpu()
    data['mkpt1'] = mat1.cpu()
    data['kpt0'] = tmp["keypoints0"][0].cpu()
    data['kpt1'] = tmp["keypoints1"][0].cpu()
    data['img_save0'] = data["img0"].squeeze(0).permute(1, 2, 0).cpu()
    data['img_save1'] = data["img1"].squeeze(0).permute(1, 2, 0).cpu()
    compute_metrics(data)
    match_true_false = torch.tensor(data['epi_errs'][0]<5e-4)

    # breakpoint()
    
    
    save_matchimg(data, f"{SAVE_PATH}/{data['identifiers'][0]}")
    # desc0, desc1, m0, match_true_false
    save_values = [m0.cpu(), match_true_false]

    return save_values




def main():
    parser = ArgumentParser(description="Testing script parameters")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser._model_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train/outputs/"+\
                        "pairs_7scenes_stairs_imrate:1_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6_rgb_UseTrueRender"
    args = get_args(parser)
    args.iteration = 30000
    model_param = Model_param.extract(args)
    pi_param = Pipe_param.extract(args)
    gaussians = GaussianModel(model_param.sh_degree)
    scene = Scene(model_param, gaussians, load_iteration=args.iteration, shuffle=False, load_feature=False, load_testcam=0)
    cameras = scene.getTrainCameras()
    leng = len(cameras)

    cnt = 0
    # all_values = []
    idx0_list = []
    idx1_list = []
    desc0_list = []
    desc1_list = []
    m0_list = []
    match_tf_list = []
    for i in range(0, leng, 10):
        for j in range(i+5, leng, 15):
            id0 = i
            id1 = j
            cam0 = cameras[id0]
            cam1 = cameras[id1]

            K = torch.tensor((cam0.intrinsic_matrix).astype(np.float32))
            T0 = cam0.extrinsic_matrix
            T1 = cam1.extrinsic_matrix
            img0 = cam0.original_image
            img1 = cam1.original_image

            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)
            T_1to0 = T_0to1.inverse()
            data = {
                "idx0": id0, 
                "idx1": id1,
                "img0": img0.unsqueeze(0).cuda(),
                "img1": img1.unsqueeze(0).cuda(),
                "K0": K,
                "K1": K,
                "T_0to1": T_0to1,
                "T_1to0": T_1to0,
                "identifiers": [f"{cam0.image_name}_{cam1.image_name}"],
            }
            cnt = cnt+1
            print(cnt)
            save_values = find_match(data)
            idx0_list.append(data["idx0"])
            idx1_list.append(data["idx1"])
            desc0_list.append(data["desc0"])
            desc1_list.append(data["desc1"])
            m0_list.append(save_values[0])
            match_tf_list.append(save_values[1])
            # breakpoint()
    all_values = {
        "idx0": idx0_list,
        "idx1": idx1_list,
        "desc0": desc0_list,
        "desc1": desc1_list,
        "m0": m0_list,
        "match_true_false": match_tf_list
    }
    torch.save(all_values, "all_values.pt")
    print(cnt)
    # breakpoint()



def test():
    USE_MLP = True
    # USE_MLP = False
    parser = ArgumentParser(description="Testing script parameters")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser._model_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train/outputs/"+\
                        "pairs_7scenes_stairs_imrate:1_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6_rgb_UseTrueRender"
    args = get_args(parser)
    args.iteration = 30000
    model_param = Model_param.extract(args)
    pi_param = Pipe_param.extract(args)
    gaussians = GaussianModel(model_param.sh_degree)
    scene = Scene(model_param, gaussians, load_iteration=args.iteration, shuffle=False, load_feature=False, load_testcam=0)
    cameras = scene.getTrainCameras()
    leng = len(cameras)


    index0 = [10, 30, 40]
    index1 = [1000, 570, 1550]

    for id0 in index0:
        for id1 in index1:
            cam0 = cameras[id0]
            cam1 = cameras[id1]
            K = torch.tensor((cam0.intrinsic_matrix).astype(np.float32))
            T0 = cam0.extrinsic_matrix
            T1 = cam1.extrinsic_matrix
            img0 = cam0.original_image
            img1 = cam1.original_image

            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)
            T_1to0 = T_0to1.inverse()
            data = {
                "idx0": id0, 
                "idx1": id1,
                "img0": img0.unsqueeze(0).cuda(),
                "img1": img1.unsqueeze(0).cuda(),
                "K0": K,
                "K1": K,
                "T_0to1": T_0to1,
                "T_1to0": T_1to0,
                "identifiers": [f"{cam0.image_name}_{cam1.image_name}"],
            }
            save_values = find_match(data, USE_MLP)


# python -m mlp.pos_neg_pair2
if __name__=="__main__":
    # main()
    # get_pos_pairs()
    test()
