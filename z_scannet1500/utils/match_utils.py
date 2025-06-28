import os
import torch
import pprint
import pickle
import numpy as np
from PIL import Image
from copy import deepcopy
from itertools import chain
from utils.match.comm import gather
from dataclasses import dataclass
from utils.match.metrics import aggregate_metrics
from utils.match.metrics_match import compute_metrics
from matchers.lightglue import LightGlue
from z_scannet1500.utils.utils import print_eval_to_file, save_matchimg_th
from utils.match.match_img import score_feature_match
from scene.colmap_loader import read_intrinsics_binary
from mlp.mlp import get_mlp_model

ROOT_PATH = "/home/koki/code/cc/feature_3dgs_2/data/img_match"
SP_THRESHOLD = 0.01
conf = {
    "sparse_outputs": True,
    "dense_outputs": True,
    "max_num_keypoints": 1024,
    "detection_threshold": SP_THRESHOLD #0.01,
}

def flattenList(x):
    return list(chain(*x))


def dump_pair():
    path = "/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test_1500_info/test.npz"
    x = dict(np.load(path))
    z=x['name']
    pair_dict = {}
    leng = int(len(z)/15)
    for i in range(leng):
        pairs = z[15*i: 15*(i+1), 2:]
        pair_dict[int(z[15 *i, 0])-707] = pairs
    intrin = dict(np.load("/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test_1500_info/intrinsics.npz"))


def read_mat_txt(path):
    with open(path, 'r') as file:
        data = file.read().split()
        data = [float(f) for f in data]
        mat = np.array(data).reshape((4,4))
    return mat


def c2w_to_w2c(T_cw):
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    R_inv = R.T        
    t_inv = -R_inv @ t
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_inv
    T_wc[:3, 3] = t_inv
    return T_wc


def match_eval(args):
    LG_THRESHOLD = 0.01
    out_name = f"{args.out_name}"
    scene_path = '/'.join((args.input).split('/')[:-1])
    scene_num = int((args.input).split('/')[-2][5:9])-707

    if args.method.startswith("SP"):
        weight = "superpoint"
        input_dim = 256
    elif args.method == "DISK":
        weight = "disk"
        input_dim = 128
    elif args.method == "ALIKED":
        weight = "aliked"
        input_dim = 128
        LG_THRESHOLD = 0.0
    matcher1 = LightGlue({
        "filter_threshold": LG_THRESHOLD ,#0.01,
        "weights": weight,
        "input_dim": input_dim,
    }).to("cuda").eval()
    
    matchers = [matcher1]
    with open(F'{ROOT_PATH}/pairs.pkl', 'rb') as f:
        my_dict = pickle.load(f)

    scene_out = f"{scene_path}/sfm_sample/outputs/{out_name}/rendering/pairs/ours_8000"
    scene_pair = f"{scene_path}/test_pairs/pose"
    intrin = read_intrinsics_binary(f"{scene_path}/sfm_sample/sparse/0/cameras.bin")[1]
    K = np.zeros((3, 3))
    K[0, 0] = intrin.params[0]
    K[1, 1] = intrin.params[1]
    K[0, 2] = intrin.params[2]
    K[1, 2] = intrin.params[3]
    K[2, 2] = 1.
    K = torch.tensor(K).float()

    if args.resize_num==2:
        K[:2, :] = K[:2, :] * 0.5

    def read_pose(scene_pair):
        poses = {}
        pose_paths = os.listdir(scene_pair)
        pose_paths = [os.path.join(scene_pair, p) for p in pose_paths]
        for pose_p in pose_paths:
            pose = read_mat_txt(pose_p)
            poses[int(pose_p.split('/')[-1].split('.')[0])] = pose        
        return poses
    
    aggregate_list = []
    mlp = get_mlp_model(args.mlp_dim, type=args.method).to("cuda")
    for match_idx, matcher in enumerate(matchers):
        if match_idx==0:
            match_result = f"{scene_path}/sfm_sample/outputs/{out_name}/{args.match_name}/LG"
            os.makedirs(f"{match_result}/images", exist_ok=True)
        else:
            match_result = f"{scene_path}/sfm_sample/outputs/{out_name}/{args.match_name}/SG"
            os.makedirs(f"{match_result}/images", exist_ok=True)
        poses = read_pose(scene_pair)
        pairs = my_dict[scene_num]
        leng = len(pairs)
        txt_file = f"{match_result}/out.txt"
        if os.path.exists(txt_file):
            os.remove(txt_file)

        for idx in range(leng):
            data = {}
            pair = pairs[idx]
            T0 = c2w_to_w2c(poses[int(pair[0])])
            T1 = c2w_to_w2c(poses[int(pair[1])])
            img0 = np.array(Image.open(f"{scene_out}/image_renders/{pair[0]}.png"))
            img1 = np.array(Image.open(f"{scene_out}/image_renders/{pair[1]}.png"))

            s0 = torch.load(f"{scene_out}/score_tensors/{pair[0]}_smap.pt").float()
            s1 = torch.load(f"{scene_out}/score_tensors/{pair[1]}_smap.pt").float()
            f0 = torch.load(f"{scene_out}/feature_tensors/{pair[0]}_fmap.pt").float()
            f1 = torch.load(f"{scene_out}/feature_tensors/{pair[1]}_fmap.pt").float()

            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)
            T_1to0 = T_0to1.inverse()
            fm_name = f"{scene_path}_score_feature_{idx}_{pair[0]}_{pair[1]}"
            data = {
                "img0": img0,
                "img1": img1,
                "s0": s0,
                "s1": s1,
                "ft0": f0,
                "ft1": f1,
                "K0": K,
                "K1": K,
                "T_0to1": T_0to1,
                "T_1to0": T_1to0,
                "identifiers": [fm_name],
            }
            data_fm = deepcopy(data)
            _ = score_feature_match(data_fm, args=args, matcher=matcher, mlp=mlp)
            fm_path = f"{match_result}/images/{idx}_score_feature_{pair[0]}_{pair[1]}.png"
            compute_metrics(data_fm)
            print_eval_to_file(data_fm, fm_name, threshold=5e-4, file_path=txt_file)
            data_fm["matcher"] = "ours+LG"
            if args.save_img:
                save_matchimg_th(data_fm, fm_path)
            keys = ['epi_errs', 'R_errs', 't_errs', 'inliers', 'identifiers']
            eval_data = {}
            for k in keys:
                eval_data[k] = data_fm[k]
            aggregate_list.append(eval_data)
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in aggregate_list]))) for k in aggregate_list[0]}
        val_metrics_4tb = aggregate_metrics(metrics, 5e-4)
        print(f"{scene_path}")
        pprint.pprint(val_metrics_4tb)
        print()
        formatted_metrics = pprint.pformat(val_metrics_4tb)
        with open(txt_file, 'a') as file:
            file.write(formatted_metrics)
        with open(f'{match_result}/matching.pkl', 'wb') as file:
            pickle.dump(aggregate_list, file)
    return aggregate_list
    

@dataclass
class Eval_params():
    out_name:str
    input: str
    method: str
    score_kpt_th: float
    kernel_size: int
    mlp_dim: int
    match_name: str
    histogram_th: float


# python eval_scannet1500.py
if __name__=="__main__":
    eval_param = Eval_params(
        out_name="SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_score0.6",
        input="/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test/scene0805_00/sfm_sample",
        method="SP",
        score_kpt_th=0.01,
        kernel_size=7,
        mlp_dim=8,
        match_name="match_test_his_0.97",
        histogram_th=0.9
    )
    match_eval(eval_param)
