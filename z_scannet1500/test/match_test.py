import os
import torch
import itertools
from tqdm import tqdm
from scene import Scene
from PIL import Image
import numpy as np
from copy import deepcopy
from z_scannet1500.utils.utils import print_eval_to_file, save_matchimg
from argparse import ArgumentParser, Namespace
from utils.metrics_match import compute_metrics
from utils.match_img import img_match2, score_feature_match
from scene.gaussian_model import GaussianModel
from utils.comm import gather
from utils.metrics import aggregate_metrics
from itertools import chain
from matchers.lightglue import LightGlue
from encoders.superpoint.superpoint import SuperPoint
import pprint

# scenes = ['0708_00', '0713_00', '0724_00']
# outputs=[0, 1, 2, 4, 5, 6]

# python eval_test.py

scenes = ['0755_00']
outputs=[2]

s_len = len(scenes)

USED_IMAGE = "image_gt"


SP_THRESHOLD = 0.01
LG_THRESHOLD = 0.01
SCORE_KPT_THRESHOLD_HIGH = 0.07
SCORE_KPT_THRESHOLD_LOW = 0.05
KERNEL_SIZE_LOW = 3
KERNEL_SIZE_HIGH = 7


def find_pairs(groups):
    all_pairs = []
    for g in groups:
        p = list(itertools.combinations(g, 2))
        all_pairs.extend(p)
    return all_pairs
    

def load_data(out_path, test_cams, n0, n1, im_name):
    cam0 = test_cams[int(n0)]
    cam1 = test_cams[int(n1)]
    img0 = np.array(Image.open(f"{out_path}/{USED_IMAGE}/{n0}.png"))
    img1 = np.array(Image.open(f"{out_path}/{USED_IMAGE}/{n1}.png"))

    H, W, _ = img0.shape
    parts = out_path.split('/')
    idx = parts.index('outputs')
    exp_idx = parts[idx+1]
    
    if exp_idx=='2' or exp_idx=='6':
        s0=s1=f0=f1=None
    else:
        sp0 = f"{out_path}/score_tensors/{n0}_smap.pt"
        sp1 = f"{out_path}/score_tensors/{n1}_smap.pt"
        s0 = torch.load(sp0)
        s1 = torch.load(sp1)
        fp0 = f"{out_path}/feature_tensors/{n0}_fmap.pt"
        fp1 = f"{out_path}/feature_tensors/{n1}_fmap.pt"
        f0 = torch.load(fp0)
        f1 = torch.load(fp1)
    K_0 = torch.tensor(cam0.intrinsic_matrix, dtype=torch.float)
    K_1 = torch.tensor(cam1.intrinsic_matrix, dtype=torch.float)
    K_0[:2, :] = K_0[:2, :] * (W/1296.)
    K_1[:2, :] = K_1[:2, :] * (H/968.)

    T0 = cam0.extrinsic_matrix
    T1 = cam1.extrinsic_matrix
    T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)
    T_1to0 = T_0to1.inverse()

    im_name = im_name+f"_{exp_idx}"
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
        "identifiers": [im_name],
        "experi_index": exp_idx,
    }
    return data


def flattenList(x):
    return list(chain(*x))


def raw_eval(args, group, out_path, m_path, test_cams, scene_num, aggregate_list):
    pairs = find_pairs(group)
    EPI_ERR_THR = 5e-4
    txt_file = f"{m_path}/out.txt"
    if os.path.exists(txt_file):
        os.remove(txt_file)
    os.makedirs(f"{m_path}/images", exist_ok=True)
    for idx, pair in enumerate(tqdm(pairs)):
        n0, n1 = pair
        im_name = f"{scene_num}_image_{idx}_{n0}_{n1}"
        data = load_data(out_path, test_cams, n0, n1, im_name)
        data_im = deepcopy(data)
        img_match2(data_im, encoder=encoder, matcher=matcher)
        compute_metrics(data_im)
        im_path = f"{m_path}/images/{idx}_image_{n0}_{n1}.png"
        print_eval_to_file(data_im, im_name, threshold=5e-4, file_path=txt_file)
        save_matchimg(data_im, im_path)
        keys = ['epi_errs', 'R_errs', 't_errs', 'inliers', 'identifiers']
        eval_data = {}
        for k in keys:
            eval_data[k] = data_im[k]
        aggregate_list.append(eval_data)
    
    metrics = {k: flattenList(gather(flattenList([_me[k] for _me in aggregate_list]))) for k in aggregate_list[0]}
    val_metrics_4tb = aggregate_metrics(metrics, EPI_ERR_THR)
    print(f"{scene_num}")
    pprint.pformat(val_metrics_4tb)
    print()
    formatted_metrics = pprint.pformat(val_metrics_4tb)
    with open(txt_file, 'a') as file:
        file.write(formatted_metrics)


def feature_eval(args, group, out_path, m_path, test_cams, scene_num, aggregate_list):
    pairs = find_pairs(group)
    EPI_ERR_THR = 5e-4
    txt_file = f"{m_path}/out.txt"
    if os.path.exists(txt_file):
        os.remove(txt_file)
    os.makedirs(f"{m_path}/images", exist_ok=True)
    for idx, pair in enumerate(tqdm(pairs)):
        n0, n1 = pair
        fm_name = f"{scene_num}_score_feature_{idx}_{n0}_{n1}"
        data = load_data(out_path, test_cams, n0, n1, im_name=fm_name)
        data_fm = deepcopy(data)
        kpt_exist = score_feature_match(data_fm, args=args, matcher=matcher, mlp_dim=16)
        if kpt_exist:
            compute_metrics(data_fm)
            fm_path = f"{m_path}/images/{idx}_score_feature_{n0}_{n1}.png"
            print_eval_to_file(data_fm, fm_name, threshold=EPI_ERR_THR, file_path=txt_file)
            save_matchimg(data_fm, fm_path)
            keys = ['epi_errs', 'R_errs', 't_errs', 'inliers', 'identifiers']
            eval_data = {}
            for k in keys:
                eval_data[k] = data_fm[k]
            aggregate_list.append(eval_data)
    metrics = {k: flattenList(gather(flattenList([_me[k] for _me in aggregate_list]))) for k in aggregate_list[0]}
    val_metrics_4tb = aggregate_metrics(metrics, EPI_ERR_THR)
    print(f"{scene_num}")
    pprint.pformat(val_metrics_4tb)
    print()

    formatted_metrics = pprint.pformat(val_metrics_4tb)
    with open(txt_file, 'a') as file:
        file.write(formatted_metrics)


matcher = LightGlue({
            "filter_threshold": LG_THRESHOLD #0.01,
        }).to("cuda").eval()

conf = {
    "sparse_outputs": True,
    "dense_outputs": True,
    "max_num_keypoints": 1024,
    "detection_threshold": SP_THRESHOLD #0.01,
}
encoder = SuperPoint(conf).to("cuda").eval()
NAME = f"sp:{SP_THRESHOLD}_lg:{LG_THRESHOLD}_kpt(h):{SCORE_KPT_THRESHOLD_HIGH}_kpt(l):{SCORE_KPT_THRESHOLD_LOW}_kernal(h):{KERNEL_SIZE_HIGH}__kernal(L):{KERNEL_SIZE_LOW}_output:{str(outputs)}_scene:{str(scenes)}"

over_all_result = f'auc_{NAME}.txt'
if os.path.exists(over_all_result):
    os.remove(over_all_result)


for out in outputs:
    aggregate_list = []
    for i in range(s_len):
        scene_num = scenes[i]
        txt = f'/home/koki/code/cc/feature_3dgs_2/{scene_num}_A.txt'
        # gp = GroupParams()
        source_path = f"/home/koki/code/cc/feature_3dgs_2/all_data/scene{scene_num}/A"
        img_path = f"{source_path}/outputs/{out}"
        model_path = f"{source_path}/outputs/{out}"
        cfgfile_string = "Namespace()"

        cfgfilepath = os.path.join(model_path, "cfg_args")
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()

        args_cfgfile = eval(cfgfile_string)
        merged_dict = vars(args_cfgfile).copy()
        merged_dict['model_path'] = model_path
        args = Namespace(**merged_dict)

        args.images = f"{args.images}"
        if hasattr(args, "foundation_model"):
            args.foundation_model = f"{args.foundation_model}"
        args.score_kpt_th_high = SCORE_KPT_THRESHOLD_HIGH
        args.score_kpt_th_low = SCORE_KPT_THRESHOLD_LOW
        args.kernel_size_low = KERNEL_SIZE_LOW
        args.kernel_size_high = KERNEL_SIZE_HIGH
        with open(txt) as file:
            group = [line.strip().split() for line in file]

        # will not use gaussian
        if out==2 or out==6:
            args.images = 'all_images/img648x484_feature'
            args.foundation_model = 'features/img648x484_feature'
            args.speedup=False
        
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, shuffle=False)
        test_cams = scene.getTestCameras()

        out_path = f"/home/koki/code/cc/feature_3dgs_2/all_data/scene{scene_num}/A/outputs/{out}/rendering/test/ours_10000"
        m_path = out_path+f"/w_match_{NAME}"
        os.makedirs(m_path, exist_ok=True)

        tmp_aggregate_list = []
        with torch.no_grad(): 
            if out==2 or out==6:
                raw_eval(args, group, out_path, m_path, test_cams, scene_num, tmp_aggregate_list)
            else:
                feature_eval(args, group, out_path, m_path, test_cams, scene_num, tmp_aggregate_list)
        
        aggregate_list.extend(tmp_aggregate_list)    
    EPI_ERR_THR = 5e-4
    metrics = {k: flattenList(gather(flattenList([_me[k] for _me in aggregate_list]))) for k in aggregate_list[0]}
    val_metrics_4tb = aggregate_metrics(metrics, EPI_ERR_THR)
    print(f"{scene_num}")
    pprint.pformat(val_metrics_4tb)
    print()

    formatted_metrics = pprint.pformat(val_metrics_4tb)
    with open(over_all_result, 'a') as file:
        file.write(f'{out} auc\n')
        file.write(formatted_metrics)
        file.write(f'\n\n')
