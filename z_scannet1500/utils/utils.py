import os
import io
import cv2
import h5py
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from scene import Scene
from copy import deepcopy
import matplotlib.pyplot as plt
from encoders.utils import GroupParams
from scene.gaussian_model import GaussianModel
from utils.metrics_match import compute_metrics
from utils.match_img import score_feature_match
from utils.viz2d import plot_image_grid, plot_keypoints, plot_matches3
# given a source path, feature path, model output path, compute some pairs of images for matching
# 2 pipelines
# img + sp + LG      -> match
# score,feature + LG -> match

# 3 to do
# 1. save output images
# 2. compute relative accuracy, save
# 3. do speed comparison
def load_pair_data(train_cams, n0, n1, gp: GroupParams, Render_type: str):
    out0 = f"{n0:05d}"
    out1 = f"{n1:05d}"
    cam0 = train_cams[n0]
    cam1 = train_cams[n1]
    img0 = Image.open(f"{gp.model_path}/{Render_type}/image_gt/{out0}.png")
    img1 = Image.open(f"{gp.model_path}/{Render_type}/image_gt/{out1}.png")
    img0 = np.array(img0)
    img1 = np.array(img1)
    # img_orig0 = Image.open(f"{gp.model_path}/{Render_type}/image_orig/{out0}.png")
    # img_orig1 = Image.open(f"{gp.model_path}/{Render_type}/image_orig/{out1}.png")
    # img_orig0 = np.array(img_orig0)
    # img_orig1 = np.array(img_orig1)
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
        if data['epi_errs'][0].size > 0:
            f.write(f"epi_err max: {data['epi_errs'][0].max()}\n")
            epi_good = data['epi_errs'][0] < threshold
            f.write(f"epi_err ratio: {epi_good.sum()/len(epi_good)}\n")
            f.write(f"epi_err ratio: {epi_good.sum()}/{len(epi_good)}\n")
        else:
            f.write("epi_err max: No data (empty array)\n")
        f.write("\n\n")


def save_matchimg(data, path, threshold=1e-4):
    all_images, all_keypoints, all_matches = [], [], []
    if isinstance(data["img0"], np.ndarray):
        all_images.append([data["img0"], data["img1"]])
    if isinstance(data["img0"], torch.Tensor):
        all_images.append([data["img_save0"], data["img_save1"]])
    all_matches.append((data['mkpt0'].cpu(), data['mkpt1'].cpu()))
    epi_good = data['epi_errs'][0] < threshold
    precision = (epi_good.sum()/len(epi_good))*100
    R_err = data['R_errs'][0]
    t_err = data['t_errs'][0]
    match_num = len(data['mkpt0'])
    fig, axes = plot_image_grid(all_images, return_fig=True, set_lim=True)
    if data.get('kpt0') is not None:
        all_keypoints.append([data['kpt0'].cpu(), data['kpt1'].cpu()])
        plot_keypoints(all_keypoints[0], axes=axes[0], colors="royalblue")
    plot_matches3(*all_matches[0], color=None, axes=axes[0], alpha=0.5, line_width=0.8, 
                  point_size=0.0, labels=epi_good,
                  captions=[data['matcher'], 
                            f"Matches: {match_num}",
                            f"Precision({threshold:.1e})({precision:.1f}%): {epi_good.sum()}/{len(epi_good)}",
                            f"R_err={R_err:.2f}, t_err={t_err:.2f}"])
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def load_array_from_s3(path, client, cv_type, use_h5py=False,):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex
    assert data is not None
    return data


def imread_gray(path, augment_fn=None, client=None):
    cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
                else cv2.IMREAD_COLOR
    if str(path).startswith('s3://'):
        image = load_array_from_s3(str(path), client, cv_type)
    else:
        image = cv2.imread(str(path), cv_type)
    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def read_scannet_gray(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    image = cv2.resize(image, resize)
    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image


# python eval.py
if __name__=="__main__":
    gp = GroupParams()
    gp.source_path = f"/home/koki/code/cc/feature_3dgs_2/all_data/scene0708_00/A"
    gp.images = "img648x484_feature"
    gp.foundation_model = "img648x484_feature"
    gp.model_path = f"{gp.source_path}/outputs/0"
    gp.eval=True

    num_pairs = 100
    mlp_dim = 16

    TYPE = "trains"
    iteration = 10000
    Render_type = f"rendering/{TYPE}/ours_{iteration}"
    parts = gp.source_path.split('/')
    scene_name = '_'.join(parts[-2:])
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
    for idx, pair in enumerate(tqdm(pairs)):
        n0, n1 = pair
        data = load_pair_data(train_cams, n0 ,n1, gp, Render_type)
        data_fm = deepcopy(data)
        score_feature_match(data_fm, mlp_dim)
        compute_metrics(data_fm)
        fm_path = f"{match_img_path}/images/{idx}_score_feature_{n0}_{n1}.png"
        fm_name = f"{scene_name}_score_feature_{idx}_{n0}_{n1}"
        print_eval_to_file(data_fm, fm_name, threshold=5e-4, file_path=txt_file)
        save_matchimg(data_fm, fm_path)
        R_err_fm_total+=data_fm['R_errs'][0]
        t_err_fm_total+=data_fm['t_errs'][0]

    with open(txt_file, 'a') as f:
        f.write(f"average feature match R_err: {R_err_fm_total/len(pairs)}\n")
        f.write(f"average feature match t_err: {t_err_fm_total/len(pairs)}\n")
