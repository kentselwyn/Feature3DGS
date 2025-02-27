import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
from copy import deepcopy
from dataclasses import dataclass
from utils.metrics_match import compute_metrics
from scene.colmap_loader import read_intrinsics_binary
from eval.eval import save_matchimg, read_scannet_gray
from encoders.superpoint.lightglue import LightGlue
from encoders.superpoint.superpoint import SuperPoint
from matchers.ASpanFormer.aspanformer import ASpanFormer
from eval.eval_scannet1500 import read_mat_txt, c2w_to_w2c
from utils.match_img import score_feature_match, encoder_img_match, semi_img_match


def get_aspan(name = "outdoor"):
    from matchers.ASpanFormer.default import get_cfg_defaults, lower_config
    config = get_cfg_defaults()
    config.merge_from_file(f"/home/koki/code/cc/feature_3dgs_2/matchers/ASpanFormer/configs/{name}/aspan_test.py")
    _config = lower_config(config)

    aspan = ASpanFormer(config=_config['aspan'])
    state_dict = torch.load(f"/home/koki/code/cc/image_matching/ml-aspanformer/weights/{name}.ckpt", 
                            map_location='cpu')['state_dict']
    aspan.load_state_dict(state_dict,strict=False)
    aspan.cuda().eval()

    return aspan


@dataclass
class Eval_params():
    score_kpt_th: float
    kernel_size: int
    mlp_dim: int
    histogram_th: float
    method: str


def read_pose(scene_pair):
    poses = {}
    pose_paths = os.listdir(scene_pair)
    pose_paths = [os.path.join(scene_pair, p) for p in pose_paths]
    for pose_p in pose_paths:
        pose = read_mat_txt(pose_p)
        poses[int(pose_p.split('/')[-1].split('.')[0])] = pose
    return poses




def read_data(scene_out, pair, intrin, poses):
    K = np.zeros((3, 3))
    K[0, 0] = intrin.params[0]
    K[1, 1] = intrin.params[1]
    K[0, 2] = intrin.params[2]
    K[1, 2] = intrin.params[3]
    K[2, 2] = 1.
    K = torch.tensor(K).float()

    T0 = c2w_to_w2c(poses[int(pair[0])])
    T1 = c2w_to_w2c(poses[int(pair[1])])
    img0 = np.array(Image.open(f"{scene_out}/image_renders/{pair[0]}.png"))
    img1 = np.array(Image.open(f"{scene_out}/image_renders/{pair[1]}.png"))


    sp0 = f"{scene_out}/score_tensors/{pair[0]}_smap.pt"
    if os.path.exists(sp0):
        s0 = torch.load(sp0).float()
        s1 = torch.load(f"{scene_out}/score_tensors/{pair[1]}_smap.pt").float()
        f0 = torch.load(f"{scene_out}/feature_tensors/{pair[0]}_fmap.pt").float()
        f1 = torch.load(f"{scene_out}/feature_tensors/{pair[1]}_fmap.pt").float()
    else:
        s0, s1, f0, f1 = None, None, None, None

    T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)
    T_1to0 = T_0to1.inverse()
    data = {
        "img0": img0,
        "img1": img1,
        "img_g0": cv2.imread(f"{scene_out}/image_gt/{pair[0]}.png", 0),
        "img_g1": cv2.imread(f"{scene_out}/image_gt/{pair[1]}.png", 0),
        "s0": s0,
        "s1": s1,
        "ft0": f0,
        "ft1": f1,
        "K0": K,
        "K1": K,
        "T_0to1": T_0to1,
        "T_1to0": T_1to0,
    }
    return data


def read_data_aspan(scene_out, pair, intrin, poses):
    # name = "image_renders"
    name = "image_renders"
    K = np.zeros((3, 3))
    K[0, 0] = intrin.params[0]
    K[1, 1] = intrin.params[1]
    K[0, 2] = intrin.params[2]
    K[1, 2] = intrin.params[3]
    K[2, 2] = 1.
    K = torch.tensor(K).float()

    T0 = c2w_to_w2c(poses[int(pair[0])])
    T1 = c2w_to_w2c(poses[int(pair[1])])
    img0 = read_scannet_gray(f"{scene_out}/{name}/{pair[0]}.png", resize=(640, 480), augment_fn=None)
    img1 = read_scannet_gray(f"{scene_out}/{name}/{pair[1]}.png", resize=(640, 480), augment_fn=None)

    s_x = 640 / 1296.
    s_y = 480 / 968.
    K_updated = deepcopy(K)
    K_updated[0, 0] *= s_x  # Update f_x
    K_updated[1, 1] *= s_y  # Update f_y
    K_updated[0, 2] *= s_x  # Update c_x
    K_updated[1, 2] *= s_y  # Update c_y

    T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)
    T_1to0 = T_0to1.inverse()
    

    data = {
        "img0": img0[None].cuda(),
        "img1": img1[None].cuda(),
        "img_save0": np.array(Image.open(f"{scene_out}/{name}/{pair[0]}.png").resize((640, 480), Image.ANTIALIAS)),
        "img_save1": np.array(Image.open(f"{scene_out}/{name}/{pair[1]}.png").resize((640, 480), Image.ANTIALIAS)),
        # "img_g0": cv2.imread(f"{scene_out}/image_renders/{pair[0]}.png", 0),
        # "img_g1": cv2.imread(f"{scene_out}/image_renders/{pair[1]}.png", 0),
        "K0": K_updated,
        "K1": K_updated,
        "T_0to1": T_0to1,
        "T_1to0": T_1to0,
    }
    return data






ROOT_PATH = "/home/koki/code/cc/feature_3dgs_2/img_match"
OUT_PATH = "/home/koki/code/cc/feature_3dgs_2/z_test_all/test"




def score_match_image(scene_name, pair_index,
                out_name = "SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6"):
    matcher = LightGlue({
                "filter_threshold": 0.01 ,
            }).to("cuda").eval()
    out_path = f"/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test/{scene_name}/sfm_sample/outputs"

    args = Eval_params(score_kpt_th=0.02, kernel_size=7, mlp_dim=16, 
                       histogram_th=None,
                    #    histogram_th=0.98, 
                       method="SP")
    scene_path = '/'.join((out_path).split('/')[:-2])
    
    
    with open(F'{ROOT_PATH}/pairs.pkl', 'rb') as f:
        my_dict = pickle.load(f)
    
    scene_num = int((out_path).split('/')[-3][5:9])-707
    pairs = my_dict[scene_num]
    pair = pairs[pair_index]
    
    scene_pair = f"{scene_path}/test_pairs/pose"
    poses = read_pose(scene_pair)
    intrin = read_intrinsics_binary(f"{scene_path}/sfm_sample/sparse/0/cameras.bin")[1]
    scene_out = f"{out_path}/{out_name}/rendering/pairs/ours_8000"
    
    data_fm = deepcopy(read_data(scene_out, pair, intrin, poses))
    _ = score_feature_match(data_fm, args=args, matcher=matcher)
    fm_path = f"{OUT_PATH}/{scene_name}_{pair_index}_score_feature_{pair[0]}_{pair[1]}.png"
    compute_metrics(data_fm)
    data_fm["matcher"] = "ours+LG"
    save_matchimg(data_fm, fm_path)
    



def sp_image_match_image(encoder, matcher, scene_name, pair_index, out_name="raw_imrate:1"):
    
    out_path = f"/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test/{scene_name}/sfm_sample/outputs"
    scene_path = '/'.join((out_path).split('/')[:-2])
    with open(F'{ROOT_PATH}/pairs.pkl', 'rb') as f:
        my_dict = pickle.load(f)
    
    scene_num = int((out_path).split('/')[-3][5:9])-707
    pairs = my_dict[scene_num]
    pair = pairs[pair_index]
    scene_pair = f"{scene_path}/test_pairs/pose"
    poses = read_pose(scene_pair)
    intrin = read_intrinsics_binary(f"{scene_path}/sfm_sample/sparse/0/cameras.bin")[1]
    scene_out = f"{out_path}/{out_name}/rendering/pairs/ours_8000"
    
    if matcher.__class__.__name__=="LightGlue":
        matcher_name = "SP+LG"
        data_fm = deepcopy(read_data(scene_out, pair, intrin, poses))
        encoder_img_match(data_fm, encoder=encoder, matcher=matcher)
    if matcher.__class__.__name__=="ASpanFormer":
        matcher_name = "ASpanFormer"
        data_fm = deepcopy(read_data_aspan(scene_out, pair, intrin, poses))
        semi_img_match(data_fm, matcher=matcher)

    compute_metrics(data_fm)
    data_fm["matcher"] = matcher_name
    fm_path = f"{OUT_PATH}/{scene_name}_{pair_index}_{matcher_name}_{pair[0]}_{pair[1]}.png"
    
    save_matchimg(data_fm, fm_path)



"/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test_1500_renders/scene0728_00/color/420.png"
"/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test_1500_renders/scene0728_00/color/975.png"

"/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test/scene0753_00/sfm_sample/outputs/SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6/rendering/pairs/ours_8000/image_renders/1320.png"
"/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test/scene0753_00/sfm_sample/outputs/SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6/rendering/pairs/ours_8000/image_renders/1440.png"

encoder = SuperPoint({
            "sparse_outputs": True,
            "max_num_keypoints": 1024,
            "detection_threshold": 0.0005,
        }).to("cuda").eval()

lg = LightGlue({
            "filter_threshold": 0.01 ,
        }).to("cuda").eval()


asapn = get_aspan(name="indoor")



# python visual.py
if __name__=="__main__":
    scene_name = "scene0783_00"
    pair_idx = 11
    # ours
    score_match_image(scene_name=scene_name, pair_index=pair_idx)
    
    # ASpanFormer
    sp_image_match_image(encoder=None, matcher=asapn, scene_name=scene_name, pair_index=pair_idx)

    # sp+lg
    sp_image_match_image(encoder=encoder, matcher=lg, scene_name=scene_name, pair_index=pair_idx)

    


