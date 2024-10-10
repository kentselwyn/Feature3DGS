import os
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from scene import Scene
import matplotlib.pyplot as plt
from gt_generation import GroupParams
from scene.gaussian_model import GaussianModel
from match_images import img_match, score_feature_match
from codes.used_codes.viz2d import plot_image_grid, plot_keypoints, plot_matches
from copy import deepcopy
from codes.metrics_match import compute_metrics
# given a source path, feature path, model output path, compute some pairs of images for matching
# 2 pipelines
# img + sp + LG      -> match
# score,feature + LG -> match

# 3 to do
# 1. save output images
# 2. compute relative accuracy, save
# 3. do speed comparison




def load_pair_data(train_cams, scene_name, n0, n1, gp: GroupParams, Render_type: str):
    out0 = f"{n0:05d}"
    out1 = f"{n1:05d}"

    cam0 = train_cams[n0]
    cam1 = train_cams[n1]

    img0 = Image.open(f"{gp.model_path}/{Render_type}/image_gt/{out0}.png")
    img1 = Image.open(f"{gp.model_path}/{Render_type}/image_gt/{out1}.png")
    img0 = np.array(img0)
    img1 = np.array(img1)

    img_orig0 = Image.open(f"{gp.model_path}/{Render_type}/image_orig/{out0}.png")
    img_orig1 = Image.open(f"{gp.model_path}/{Render_type}/image_orig/{out1}.png")
    img_orig0 = np.array(img_orig0)
    img_orig1 = np.array(img_orig1)

    H, W, _ = img0.shape

    sp0 = f"{gp.model_path}/{Render_type}/score_tensors/{out0}_smap_CxHxW.pt"
    sp1 = f"{gp.model_path}/{Render_type}/score_tensors/{out1}_smap_CxHxW.pt"
    s0 = torch.load(sp0)
    s1 = torch.load(sp1)

    fp0 = f"{gp.model_path}/{Render_type}/feature_tensors/{out0}_fmap_CxHxW.pt"
    fp1 = f"{gp.model_path}/{Render_type}/feature_tensors/{out1}_fmap_CxHxW.pt"
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

    draw_namw = f"{scene_name[:-2]}_n0_n1"
    data = {
        "img0": img0,
        "img1": img1,
        "img_orig0": img_orig0,
        "img_orig1": img_orig1,
        "s0": s0,
        "s1": s1,
        "ft0": f0,
        "ft1": f1,
        "K0": K_0,
        "K1": K_1,
        "T_0to1": T_0to1,
        "T_1to0": T_1to0,
        "name": draw_namw,
    }
    return data



def print_eval(data, name, threshold=5e-4):
    print(f"======{name}======")
    print("R_err: ",data['R_errs'][0])
    print("t_err: ",data['t_errs'][0])
    print(f"inlier ratio: {data['inliers'][0].sum()}/{len(data['inliers'][0])}")
    print("epi_err max: ", data['epi_errs'].max())
    epi_good = data['epi_errs']<threshold
    print("epi_err ratio: ", epi_good.sum()/len(epi_good))


def print_eval_to_file(data, name, threshold=5e-4, file_path='output.txt'):
    with open(file_path, 'a') as f:  # 'a' mode for appending to the file
        f.write(f"======{name}======\n")
        f.write(f"R_err: {data['R_errs'][0]}\n")
        f.write(f"t_err: {data['t_errs'][0]}\n")
        f.write(f"inlier ratio: {data['inliers'][0].sum()}/{len(data['inliers'][0])}\n")
        f.write(f"epi_err max: {data['epi_errs'].max()}\n")
        
        epi_good = data['epi_errs'] < threshold
        f.write(f"epi_err ratio: {epi_good.sum()/len(epi_good)}\n")
        if name[:5]=="image":
            f.write("\n\n")




def save_matchimg(data, path):
    all_images, all_keypoints, all_matches = [], [], []
    all_images.append([data["img_orig0"], data["img_orig1"]])
    all_keypoints.append([data['kpt0'], data['kpt1']])
    all_matches.append((data['mkpt0'], data['mkpt1']))
    
    fig, axes = plot_image_grid(all_images, return_fig=True, set_lim=True)
    plot_keypoints(all_keypoints[0], axes=axes[0], colors="royalblue")
    plot_matches(*all_matches[0], color=None, axes=axes[0], alpha=0.5, line_width=1.0, point_size=0.0)

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



# python evaluation.py
if __name__=="__main__":
    scene_name = "scene0000_01/A"
    gp = GroupParams()
    gp.source_path = f"/home/koki/code/cc/feature_3dgs_2/all_data/{scene_name}"
    gp.foundation_model = "imrate:2_th:0.01_mlpdim:16"
    gp.model_path = f"{gp.source_path}/outputs/imrate:2_th:0.01_mlpdim:16"
    num_pairs = 100
    mlp_dim = 16
    Render_type = "rendering/trains/ours_7000"


    gaussians = GaussianModel(gp.sh_degree)
    scene = Scene(gp, gaussians, shuffle=False)
    train_cams = scene.getTrainCameras()

    img_list = list(range(len(train_cams)))


    random.seed(0)
    pairs = []
    while len(pairs) < num_pairs:
        pair = random.sample(img_list, 2)
        pairs.append(tuple(pair))

    match_img_path = f"{gp.model_path}/{Render_type}/w_match"
    txt_file = f"{match_img_path}/out.txt"
    os.makedirs(f"{match_img_path}/images", exist_ok=True)


    R_err_fm_total = 0.
    t_err_fm_total = 0.
    R_err_im_total = 0.
    t_err_im_total = 0.
    for idx, pair in enumerate(tqdm(pairs)):
        n0, n1 = pair
        data = load_pair_data(train_cams, scene_name, n0 ,n1, gp, Render_type)

        d_fm = deepcopy(data)
        d_im = deepcopy(data)

        score_feature_match(d_fm, mlp_dim)
        img_match(d_im)

        compute_metrics(d_fm)
        compute_metrics(d_im)

        fm_path = f"{match_img_path}/images/{idx}_score_feature_{n0}_{n1}.png"
        im_path = f"{match_img_path}/images/{idx}_image_{n0}_{n1}.png"
        fm_name = f"score_feature_{idx}_{n0}_{n1}"
        im_name = f"image_{idx}_{n0}_{n1}"

        print_eval_to_file(d_fm, fm_name, threshold=5e-4, file_path=txt_file)
        print_eval_to_file(d_im, im_name, threshold=5e-4, file_path=txt_file)

        save_matchimg(d_fm, fm_path)
        save_matchimg(d_im, im_path)

        R_err_fm_total+=d_fm['R_errs'][0]
        t_err_fm_total+=d_fm['t_errs'][0]

        R_err_im_total+=d_im['R_errs'][0]
        t_err_im_total+=d_im['t_errs'][0]




    with open(txt_file, 'a') as f:
        f.write(f"average feature match R_err: {R_err_fm_total/len(pairs)}\n")
        f.write(f"average feature match t_err: {t_err_fm_total/len(pairs)}\n")

        f.write(f"average image match R_err: {R_err_im_total/len(pairs)}\n")
        f.write(f"average image match t_err: {t_err_im_total/len(pairs)}\n")












