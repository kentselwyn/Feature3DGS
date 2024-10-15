import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from encoders.superpoint.mlp import get_mlp_model
from codes.used_codes.viz2d import plot_image_grid, plot_keypoints, plot_matches
import time
from typing import Tuple


def extract_kpt(score: torch.Tensor, threshold = 0.3):
    score = score[0]
    mask = (score > threshold).cpu().numpy()
    start = time.time()
    labels = label(mask)
    regions = regionprops(labels)
    centroids = [region.centroid for region in regions]
    end = time.time()

    # print("elapsed time:", end-start)
    
    centroids = torch.tensor(centroids)
    return centroids


def find_small_circle_centers(score_map, threshold, kernel_size=3):
    """
    Find the centers of small circles (2-3 pixels in diameter) in a score map using GPU acceleration.

    Args:
        score_map (torch.Tensor): The input score map tensor of shape [1, H, W] on GPU.
        threshold (float): The minimum value to consider a point as a potential center.

    Returns:
        torch.Tensor: Tensor containing the coordinates of the circle centers.
    """
    # Ensure the score map is on the GPU
    score_map = score_map.cuda()

    # Apply max pooling with a small kernel to find local maxima
    pooled = F.max_pool2d(score_map, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    maxima = (score_map == pooled) & (score_map > threshold)

    positions = torch.nonzero(maxima[0], as_tuple=False)

    return positions


def sample_descriptors_fix_sampling(kpt, desc, scale):
    c, _, _ = desc.shape
    kpt = kpt / scale
    kpt = kpt*2 - 1
    kpt = kpt.float()
    desc = desc.unsqueeze(0).float() # add batch dim
    desc = torch.nn.functional.grid_sample(desc, kpt.view(1, 1, -1, 2), mode="bilinear", align_corners=False)
    desc = desc.reshape(1, c, -1).transpose(-1,-2)

    return desc





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def match_image(img0, img1, kpt0, kpt1, desc0, desc1, mlp_dim, matcher):
    
    mlp = get_mlp_model(mlp_dim).to("cuda")
    data = {}
    kpt0 = kpt0.to(device).unsqueeze(0).float()
    kpt1 = kpt1.to(device).unsqueeze(0).float()

    data["keypoints0"] = kpt0
    data["keypoints1"] = kpt1
    data["descriptors0"] = mlp.decode(desc0.to(device)).unsqueeze(0).float()
    data["descriptors1"] = mlp.decode(desc1.to(device)).unsqueeze(0).float()
    data["image_size"] = img0.shape[:2]

    pred = matcher(data)
    
    m0 = pred['m0']
    m1 = pred['m1']
    valid = (m0[0] > -1)

    m_kpts0, m_kpts1 = kpt0[0][valid].cpu().numpy(), kpt1[0][m0[0][valid]].cpu().numpy()
    

    all_images, all_keypoints, all_matches = [], [], []
    all_images.append([img0, img1])
    all_keypoints.append([kpt0[0].to("cpu"), kpt1[0].to("cpu")])
    all_matches.append((m_kpts0, m_kpts1))


    fig, axes = plot_image_grid(all_images, return_fig=True, set_lim=True)
    plot_keypoints(all_keypoints[0], axes=axes[0], colors="royalblue")
    plot_matches(*all_matches[0], color=None, axes=axes[0], alpha=0.5, line_width=1.0, point_size=0.0)

    return fig

    

def load_kpt_desc(name, folder_path):
    i_path = f"{folder_path}/image_renders/{name}.png"
    s_path = f"{folder_path}/score_tensors/{name}_smap_CxHxW.pt"
    f_path = f"{folder_path}/feature_tensors/{name}_fmap_CxHxW.pt"

    score = torch.load(s_path)
    feat = torch.load(f_path)

    kpts = extract_kpt(score)

    # [h, w] to [w, h]
    kpts = torch.tensor(kpts)[:, [1, 0]]

    _, h, w = score.shape
    scale = torch.tensor([w, h])
    desc = sample_descriptors_fix_sampling(kpts, feat, scale)

    image = Image.open(i_path)
    image = np.array(image)

    return image, kpts, desc



def matchimg2(img0, img1, s0, s1, f0, f1, matcher, mlp_dim=16, draw_name=None) -> Tuple[torch.Tensor, torch.Tensor]:
    kpt0 = extract_kpt(s0)
    kpt1 = extract_kpt(s1)

    kpt0 = kpt0.clone().detach()[:, [1, 0]]
    kpt1 = kpt1.clone().detach()[:, [1, 0]]

    _, h, w = s0.shape
    scale = torch.tensor([w, h])
    desc0 = sample_descriptors_fix_sampling(kpt0, f0, scale)
    desc1 = sample_descriptors_fix_sampling(kpt1, f1, scale)
    mlp = get_mlp_model(mlp_dim).to("cuda")
    data = {}
    kpt0 = kpt0.to(device).unsqueeze(0).float()
    kpt1 = kpt1.to(device).unsqueeze(0).float()

    data["keypoints0"] = kpt0
    data["keypoints1"] = kpt1
    data["descriptors0"] = mlp.decode(desc0.to(device)).float()
    data["descriptors1"] = mlp.decode(desc1.to(device)).float()
    data["image_size"] = s0.shape[1:]

    pred = matcher(data)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m_kpts0, m_kpts1 = kpt0[0][valid].cpu().numpy(), kpt1[0][m0[0][valid]].cpu().numpy()


    if draw_name is not None:
        all_images, all_keypoints, all_matches = [], [], []
        all_images.append([img0, img1])
        all_keypoints.append([kpt0[0].to("cpu"), kpt1[0].to("cpu")])
        all_matches.append((m_kpts0, m_kpts1))

        fig, axes = plot_image_grid(all_images, return_fig=True, set_lim=True)
        plot_keypoints(all_keypoints[0], axes=axes[0], colors="royalblue")
        plot_matches(*all_matches[0], color=None, axes=axes[0], alpha=0.5, line_width=1.0, point_size=0.0)

        plt.savefig(f"{draw_name}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    m_kpts0 = torch.tensor(m_kpts0)
    m_kpts1 = torch.tensor(m_kpts1)

    return m_kpts0, m_kpts1







def score_feature_match(data, args, matcher, mlp_dim=16):
    
    st = time.time()
    # if data['experi_index']=='5' or data['experi_index']=='4':
    #     kpt0 = extract_kpt(data['s0'], threshold=args.score_kpt_th_low)
    #     kpt1 = extract_kpt(data['s1'], threshold=args.score_kpt_th_low)
    # else:
    #     kpt0 = extract_kpt(data['s0'], threshold=args.score_kpt_th_high)
    #     kpt1 = extract_kpt(data['s1'], threshold=args.score_kpt_th_high)
    if data['experi_index']=='5' or data['experi_index']=='4':
        kpt0 = find_small_circle_centers(data['s0'], threshold=args.score_kpt_th_low, kernel_size=args.kernel_size_low)
        kpt1 = find_small_circle_centers(data['s1'], threshold=args.score_kpt_th_low, kernel_size=args.kernel_size_low)
    else:
        kpt0 = find_small_circle_centers(data['s0'], threshold=args.score_kpt_th_high, kernel_size=args.kernel_size_high)
        kpt1 = find_small_circle_centers(data['s1'], threshold=args.score_kpt_th_high, kernel_size=args.kernel_size_high)
    en = time.time()
    # print("first: ",en-st)

    if kpt0.dim()<2:
        return False

    st = time.time()
    kpt0 = kpt0.clone().detach()[:, [1, 0]].to(data['ft0'])
    kpt1 = kpt1.clone().detach()[:, [1, 0]].to(data['ft0'])
    _, h, w = data['s0'].shape
    scale = torch.tensor([w, h]).to(kpt0)
    desc0 = sample_descriptors_fix_sampling(kpt0, data['ft0'], scale)
    desc1 = sample_descriptors_fix_sampling(kpt1, data['ft1'], scale)
    mlp = get_mlp_model(mlp_dim).to("cuda")
    en = time.time()
    # print("second: ",en-st)

    st = time.time()
    tmp = {}
    tmp["keypoints0"] = kpt0.to(device).unsqueeze(0).float()
    tmp["keypoints1"] = kpt1.to(device).unsqueeze(0).float()

    tmp["descriptors0"] = mlp.decode(desc0.to(device)).float()
    tmp["descriptors1"] = mlp.decode(desc1.to(device)).float()
    tmp["image_size"] = data['s0'].shape[1:]

    pred = matcher(tmp)
    en = time.time()
    # print("third: ",en-st)

    m0 = pred['m0']
    valid = (m0[0] > -1)

    m0, m1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[0][valid]].cpu()

    data['mkpt0'] = m0
    data['mkpt1'] = m1
    data['kpt0'] = kpt0
    data['kpt1'] = kpt1

    return True
    


def img_match(data: dict, encoder, matcher) -> Tuple[torch.Tensor, torch.Tensor]:
    d0 = {}
    d1 = {}
    transfrom = transforms.ToTensor()
    d0['image'] = transfrom(data['img0']).to("cuda").unsqueeze(0)
    d1['image'] = transfrom(data['img1']).to("cuda").unsqueeze(0)

    p0 = encoder(d0)
    p1 = encoder(d1)

    tmp = {}
    tmp["keypoints0"] = p0['keypoints']
    tmp["keypoints1"] = p1['keypoints']
    tmp["descriptors0"] = p0['descriptors']
    tmp["descriptors1"] = p1['descriptors']
    tmp["image_size"] = d0['image'][0].shape[1:]

    pred = matcher(tmp)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[0][valid]].cpu()
    kpt0, kpt1 = tmp['keypoints0'][0].cpu(), tmp['keypoints1'][0].cpu()

    data['mkpt0'] = m0
    data['mkpt1'] = m1
    data['kpt0'] = kpt0
    data['kpt1'] = kpt1


def img_match2(data: dict, encoder, matcher) -> Tuple[torch.Tensor, torch.Tensor]:
    d0 = {}
    d1 = {}
    d0['image'] = data['img0'].unsqueeze(0)
    d1['image'] = data['img1'].unsqueeze(0)

    p0 = encoder(d0)
    p1 = encoder(d1)

    tmp = {}
    tmp["keypoints0"] = p0['keypoints']
    tmp["keypoints1"] = p1['keypoints']
    tmp["descriptors0"] = p0['descriptors']
    tmp["descriptors1"] = p1['descriptors']
    tmp["image_size"] = d0['image'][0].shape[1:]

    pred = matcher(tmp)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[0][valid]].cpu()
    kpt0, kpt1 = tmp['keypoints0'][0].cpu(), tmp['keypoints1'][0].cpu()

    data['mkpt0'] = m0
    data['mkpt1'] = m1
    data['kpt0'] = kpt0
    data['kpt1'] = kpt1




# python match_images.py
if __name__=="__main__":
    # name0 = "00019"
    # name1 = "00083"
    # mlp_dim = 16
    # render_path = "/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_01/A/outputs/imrate:2_th:0.01_mlpdim:16/rendering/trains/ours_7000"
    # img0, kpt0, desc0 = load_kpt_desc(name0, folder_path=render_path)
    # img1, kpt1, desc1 = load_kpt_desc(name1, folder_path=render_path)

    # fig = match_image(img0, img1, kpt0, kpt1, desc0, desc1, mlp_dim)

    # plt.savefig(f"match_{name0}_{name1}_novel_dim{mlp_dim}.png", bbox_inches='tight', pad_inches=0)
    # plt.close(fig)

    pass

    



    