import os
import time
import torch
import pprint
import numpy as np
from PIL import Image
from copy import deepcopy
from visual import get_aspan, Eval_params
from utils.match.metrics import aggregate_metrics
from utils.match.metrics_match import compute_metrics
from matchers.lightglue import LightGlue
from encoders.superpoint.superpoint import SuperPoint
from eval.eval_scannet1500 import flattenList, gather
from eval.eval import save_matchimg, read_scannet_gray
from utils.match.match_img import score_feature_match, encoder_img_match, semi_img_match


all_path="/home/koki/code/cc/feature_3dgs_2/img_match/Else/tandt_db"
# out_name = "SP_tank_db_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_images_low_resolution"
out_name = "raw_images_low_resolution"

encoder = SuperPoint({
            "sparse_outputs": True,
            "max_num_keypoints": 1024,
            "detection_threshold": 0.005,
        }).to("cuda").eval()

lg = LightGlue({
            "filter_threshold": 0.01 ,
        }).to("cuda").eval()


asapn = get_aspan(name="indoor")




def read_data(path, idx, num, fold):
    K = torch.tensor(np.load(f"{path}/intrinsics/{idx:05d}_{num}.npy"), dtype=torch.float)
    if fold=="train" or fold=="truck":
        scaling_factor = 0.5
        K[0, 0] *= scaling_factor  # Scale f_x
        K[1, 1] *= scaling_factor  # Scale f_y
        K[0, 2] *= scaling_factor  # Scale c_x
        K[1, 2] *= scaling_factor

    T_0to1 = torch.tensor(np.load(f"{path}/extrinsics/{idx:05d}_{num}.npy"), dtype=torch.float)
    T_1to0 = T_0to1.inverse()
    img0 = np.array(Image.open(f"{path}/image_renders/{idx:05d}.png"))
    img1 = np.array(Image.open(f"{path}/image_renders/{idx:05d}_{num}.png"))

    if os.path.exists(f"{path}/score_tensors/{idx:05d}_smap.pt"):
        s0 = torch.load(f"{path}/score_tensors/{idx:05d}_smap.pt").float()
        s1 = torch.load(f"{path}/score_tensors/{idx:05d}_{num}_smap.pt").float()
        f0 = torch.load(f"{path}/feature_tensors/{idx:05d}_fmap.pt").float()
        f1 = torch.load(f"{path}/feature_tensors/{idx:05d}_{num}_fmap.pt").float()
    else:
        s0, s1, f0, f1 = None, None, None, None
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
    }
    return data


def read_data_aspan(path, idx, num, fold):
    K = torch.tensor(np.load(f"{path}/intrinsics/{idx:05d}_{num}.npy"), dtype=torch.float)
    if fold=="train" or fold=="truck":
        scaling_factor = 0.5
        K[0, 0] *= scaling_factor  # Scale f_x
        K[1, 1] *= scaling_factor  # Scale f_y
        K[0, 2] *= scaling_factor  # Scale c_x
        K[1, 2] *= scaling_factor

    T_0to1 = torch.tensor(np.load(f"{path}/extrinsics/{idx:05d}_{num}.npy"), dtype=torch.float)
    T_1to0 = T_0to1.inverse()
    img0_ = np.array(Image.open(f"{path}/image_renders/{idx:05d}.png"))
    H, W, C = img0_.shape
    # img1 = np.array(Image.open(f"{path}/image_renders/{idx:05d}_{num}.png"))
    img0 = read_scannet_gray(f"{path}/image_renders/{idx:05d}.png", resize=(640, 480), augment_fn=None)
    img1 = read_scannet_gray(f"{path}/image_renders/{idx:05d}_{num}.png", resize=(640, 480), augment_fn=None)
    s_x = 640 / float(W)
    s_y = 480 / float(H)
    K_updated = deepcopy(K)
    K_updated[0, 0] *= s_x  # Update f_x
    K_updated[1, 1] *= s_y  # Update f_y
    K_updated[0, 2] *= s_x  # Update c_x
    K_updated[1, 2] *= s_y  # Update c_y
    data = {
        "img0": img0[None].cuda(),
        "img1": img1[None].cuda(),
        "K0": K_updated,
        "K1": K_updated,
        "T_0to1": T_0to1,
        "T_1to0": T_1to0,
    }
    return data






def eval_blending_ours():
    folds = os.listdir(all_path)
    folds = [fold for fold in folds if os.path.isdir(os.path.join(all_path, fold))]
    args = Eval_params(score_kpt_th=0.02, kernel_size=7, mlp_dim=16, 
                       histogram_th=0.98,
                       method="SP_tank_db")
    aggregate_list = []
    for fold in folds:
        print(fold)
        path = f"{all_path}/{fold}/outputs/{out_name}/rendering/test_pairs/ours_8000"
        match_path = f"{path}/match_img_ours"
        os.makedirs(match_path, exist_ok=True)

        leng = int(len(os.listdir(f"{path}/extrinsics"))/2)
        for idx in range(leng):
            for num in range(1, 3):
                data_fm = deepcopy(read_data(path, idx, num, fold))
                _ = score_feature_match(data_fm, args, lg)
                fm_name = f"{fold}_match_{idx}_{num}"
                fm_path = f"{match_path}/match_{idx}_{num}.png"
                compute_metrics(data_fm)
                data_fm["matcher"] = "ours+LG"
                data_fm["identifiers"] = [fm_name]
                save_matchimg(data_fm, fm_path, threshold=1e-4)

                keys = ['epi_errs', 'R_errs', 't_errs', 'inliers', 'identifiers']
                eval_data = {}
                for k in keys:
                    eval_data[k] = data_fm[k]
                aggregate_list.append(eval_data)
    metrics = {k: flattenList(gather(flattenList([_me[k] for _me in aggregate_list]))) 
               for k in aggregate_list[0]}
    val_metrics_4tb = aggregate_metrics(metrics, 1e-4)

    pprint.pprint(val_metrics_4tb)
    formatted_metrics = pprint.pformat(val_metrics_4tb)
    txt_file = f"{all_path}/ours.txt"
    with open(txt_file, 'a') as file:
        file.write(formatted_metrics)



def eval_blending_sp_lg():
    folds = os.listdir(all_path)
    folds = [fold for fold in folds if os.path.isdir(os.path.join(all_path, fold))]
    aggregate_list = []
    for fold in folds:
        # if fold=="train":
        print(fold)
        path = f"{all_path}/{fold}/outputs/{out_name}/rendering/test_pairs/ours_8000"
        match_path = f"{path}/match_sp_lg"
        os.makedirs(match_path, exist_ok=True)
        leng = int(len(os.listdir(f"{path}/extrinsics"))/2)
        for idx in range(leng):
            for num in range(1, 3):
                data_fm = deepcopy(read_data(path, idx, num, fold))
                encoder_img_match(data_fm, encoder=encoder, matcher=lg)

                if data_fm['mkpt0'] is None:
                    continue

                fm_name = f"{fold}_match_{idx}_{num}"
                fm_path = f"{match_path}/match_{idx}_{num}.png"
                compute_metrics(data_fm)
                data_fm["matcher"] = "sp+LG"
                data_fm["identifiers"] = [fm_name]
                save_matchimg(data_fm, fm_path, threshold=1e-4)

                keys = ['epi_errs', 'R_errs', 't_errs', 'inliers', 'identifiers']
                eval_data = {}
                for k in keys:
                    eval_data[k] = data_fm[k]
                aggregate_list.append(eval_data)
    metrics = {k: flattenList(gather(flattenList([_me[k] for _me in aggregate_list]))) for k in aggregate_list[0]}
    val_metrics_4tb = aggregate_metrics(metrics, 1e-4)
    pprint.pprint(val_metrics_4tb)
    formatted_metrics = pprint.pformat(val_metrics_4tb)
    txt_file = f"{all_path}/sp_lg.txt"
    with open(txt_file, 'a') as file:
        file.write(formatted_metrics)



def eval_blending_aspan():
    folds = os.listdir(all_path)
    folds = [fold for fold in folds if os.path.isdir(os.path.join(all_path, fold))]
    folds = sorted(folds)
    
    aggregate_list = []
    for fold in folds:
        
        if fold=="train" or fold=="truck":
            asapn = get_aspan(name="outdoor")
        else:
            asapn = get_aspan(name="indoor")
        print(fold)
        path = f"{all_path}/{fold}/outputs/{out_name}/rendering/test_pairs/ours_8000"
        match_path = f"{path}/match_aspan"
        os.makedirs(match_path, exist_ok=True)
        leng = int(len(os.listdir(f"{path}/extrinsics"))/2)
        for idx in range(leng):
            for num in range(1, 3):
                data_fm = deepcopy(read_data_aspan(path, idx, num, fold))
                semi_img_match(data_fm, matcher=asapn)

                if data_fm['mkpt0'] is None:
                    continue

                fm_name = f"{fold}_match_{idx}_{num}"
                fm_path = f"{match_path}/match_{idx}_{num}.png"
                compute_metrics(data_fm)
                data_fm["matcher"] = "ASpanFormer"
                data_fm["identifiers"] = [fm_name]
                save_matchimg(data_fm, fm_path, threshold=1e-4)

                keys = ['epi_errs', 'R_errs', 't_errs', 'inliers', 'identifiers']
                eval_data = {}
                for k in keys:
                    eval_data[k] = data_fm[k]
                aggregate_list.append(eval_data)
    metrics = {k: flattenList(gather(flattenList([_me[k] for _me in aggregate_list]))) for k in aggregate_list[0]}
    val_metrics_4tb = aggregate_metrics(metrics, 1e-4)
    pprint.pprint(val_metrics_4tb)
    formatted_metrics = pprint.pformat(val_metrics_4tb)
    txt_file = f"{all_path}/aspan.txt"
    with open(txt_file, 'a') as file:
        file.write(formatted_metrics)








# python eval_blending_truck.py
if __name__=="__main__":
    start = time.time()
    # eval_blending_ours()
    eval_blending_sp_lg()
    # eval_blending_aspan()
    end = time.time()
    print("elapsed time:", end-start)
    


