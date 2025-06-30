import time
import torch
import numpy as np
from PIL import Image
from torch import nn
from typing import Tuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
from omegaconf import OmegaConf
from matchers.aliked import SDDH
from torchvision import transforms
from skimage.measure import label, regionprops
from mlp.mlp import get_mlp_model
from utils.viz2d import plot_image_grid, plot_keypoints, plot_matches,\
                                plot_matches3
from scipy.ndimage import gaussian_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_kpt(score: torch.Tensor, threshold = 0.3):
    score = score[0]
    mask = (score > threshold).cpu().numpy()
    start = time.time()
    labels = label(mask)
    regions = regionprops(labels)
    centroids = [region.centroid for region in regions]
    end = time.time()
    centroids = torch.tensor(centroids, dtype=torch.float16)
    return centroids


def render_gaussian_kpt_map(kpts: torch.Tensor, shape: tuple, sigma=2.0):
    """
    將 keypoints 轉為 2D Gaussian heatmap
    - kpts: Tensor of shape [N, 2], (y, x)
    - shape: (H, W)
    - sigma: Gaussian sigma
    Return:
    - map: Tensor of shape [1, H, W]
    """
    H, W = shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    # 對每個 keypoint 畫一個 Gaussian
    for y, x in kpts.long():
        if 0 <= y < H and 0 <= x < W:
            temp_map = np.zeros_like(heatmap)
            temp_map[y, x] = 1.0
            temp_map = gaussian_filter(temp_map, sigma=sigma)
            heatmap += temp_map

    # 正規化到 [0, 1]
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
    return heatmap


def generate_gaussian_kernel(size=7, sigma=1.0):
    """產生 2D Gaussian kernel"""
    x = torch.arange(size).float() - size // 2
    y = torch.arange(size).float() - size // 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def fast_render_gaussian_kpt_map(kpts: torch.Tensor, shape: tuple, sigma=1.5, kernel_size=7, device="cpu"):
    """
    更快速地產生 Gaussian map
    - kpts: [N, 2] (y, x)
    - shape: (H, W)
    """
    H, W = shape
    heatmap = torch.zeros((1, H, W), device=device)
    # 產生 Gaussian kernel（例如 7x7）
    kernel = generate_gaussian_kernel(kernel_size, sigma).to(device)
    offset = kernel_size // 2
    kpts = kpts.long()
    for y, x in kpts:
        if offset <= y < H - offset and offset <= x < W - offset:
            heatmap[0, y - offset: y + offset + 1, x - offset: x + offset + 1] += kernel
    # 可選：正規化到 [0, 1]
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap



def find_small_circle_centers(score_map, threshold, kernel_size=3):
    """
    Find the centers of small circles (2-3 pixels in diameter) in a score map using GPU acceleration.

    Args:
        score_map (torch.Tensor): The input score map tensor of shape [1, H, W] on GPU.
        threshold (float): The minimum value to consider a point as a potential center.

    Returns:
        torch.Tensor: Tensor containing the coordinates of the circle centers.
    """
    score_map = score_map.cuda()
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


def matchimg2(img0, img1, s0, s1, f0, f1, matcher, mlp_dim=16, draw_path=None) -> Tuple[torch.Tensor, torch.Tensor]:
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


    if draw_path is not None:
        all_images, all_keypoints, all_matches = [], [], []
        all_images.append([img0, img1])
        all_keypoints.append([kpt0[0].to("cpu"), kpt1[0].to("cpu")])
        all_matches.append((m_kpts0, m_kpts1))

        fig, axes = plot_image_grid(all_images, return_fig=True, set_lim=True)
        plot_keypoints(all_keypoints[0], axes=axes[0], colors="royalblue")
        plot_matches(*all_matches[0], color=None, axes=axes[0], alpha=0.5, line_width=1.0, point_size=0.0)

        plt.savefig(f"{draw_path}", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    m_kpts0 = torch.tensor(m_kpts0)
    m_kpts1 = torch.tensor(m_kpts1)

    return m_kpts0, m_kpts1



def choose_th(score, args):
    score_flat = score.flatten()
    percentile_value = torch.quantile(score_flat, float(args.histogram_th))
    return percentile_value.item()

# python -m utils.match_img
mlp = get_mlp_model(16, type="SP_tank_db").to("cuda")


def score_feature_match(data, args, matcher, mlp):
    st = time.time()
    if args.histogram_th is not None:
        th0 = choose_th(data['s0'], args)
        th1 = choose_th(data['s1'], args)
    else:
        th0 = args.score_kpt_th
        th1 = args.score_kpt_th
    kpt0 = find_small_circle_centers(data['s0'], threshold=th0, kernel_size=args.kernel_size)
    kpt1 = find_small_circle_centers(data['s1'], threshold=th1, kernel_size=args.kernel_size)
    en = time.time()
    if kpt0.dim()<2:
        return False
    st = time.time()
    kpt0 = kpt0.clone().detach()[:, [1, 0]].to(data['ft0'])
    kpt1 = kpt1.clone().detach()[:, [1, 0]].to(data['ft1'])
    _, h, w = data['s0'].shape
    scale = torch.tensor([w, h]).to(kpt0)
    desc0 = sample_descriptors_fix_sampling(kpt0, data['ft0'], scale)
    desc1 = sample_descriptors_fix_sampling(kpt1, data['ft1'], scale)
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
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[0][valid]].cpu()
    data['mkpt0'] = m0
    data['mkpt1'] = m1
    data['kpt0'] = kpt0
    data['kpt1'] = kpt1
    return True



config = {
    "c1": 16,
    "c2": 32,
    "c3": 64,
    "c4": 128,
    "dim": 128,
    "K": 3,
    "M": 16,
}
config = OmegaConf.create(config)

desc_head = SDDH(config.dim, config.K, config.M, gate=nn.SELU(inplace=True), conv2D=False, mask=False)
state_dict = torch.hub.load_state_dict_from_url(
                "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth".format("aliked-n16"), map_location="cpu"
            )
desc_head_state_dict = {k.replace('desc_head.', ''): v for k, v in state_dict.items() if k.startswith('desc_head')}
desc_head.load_state_dict(desc_head_state_dict)
desc_head = desc_head.to(device)



def score_feature_aliked(data, args, matcher):
    kpt0 = find_small_circle_centers(data['s0'], threshold=args.score_kpt_th, kernel_size=args.kernel_size)
    kpt1 = find_small_circle_centers(data['s1'], threshold=args.score_kpt_th, kernel_size=args.kernel_size)
    kpt0 = kpt0.clone().detach()[:, [1, 0]].to(device)
    kpt1 = kpt1.clone().detach()[:, [1, 0]].to(device)
    if kpt0.dim()<2:
        return False
    kpt0 = kpt0.clone().detach()[:, [1, 0]].to(device).float()
    kpt1 = kpt1.clone().detach()[:, [1, 0]].to(device).float()
    _, h, w = data['s0'].shape
    scale = torch.tensor([w, h]).to(device)
    desc0 = sample_descriptors_fix_sampling(kpt0, data['ft0'].to(device), scale)
    desc1 = sample_descriptors_fix_sampling(kpt1, data['ft1'].to(device), scale)
    mlp = get_mlp_model(args.mlp_dim, type=args.method).to(kpt0)
    tmp = {}
    tmp["keypoints0"] = kpt0.to(device).unsqueeze(0).float()
    tmp["keypoints1"] = kpt1.to(device).unsqueeze(0).float()
    tmp["descriptors0"] = mlp.decode(desc0.to(device)).float()
    tmp["descriptors1"] = mlp.decode(desc1.to(device)).float()
    tmp["image_size"] = data['s0'].shape[1:]
    pred = matcher(tmp)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[0][valid]].cpu()
    data['mkpt0'] = m0
    data['mkpt1'] = m1
    data['kpt0'] = kpt0
    data['kpt1'] = kpt1
    return True


def score_feature_one(data, matcher, threshold=0.1):
    kpt0 = find_small_circle_centers(data['s0'], threshold=threshold, kernel_size=7)
    kpt1 = find_small_circle_centers(data['s1'], threshold=threshold, kernel_size=7)
    kpt0 = kpt0.clone().detach()[:, [1, 0]].to(data['ft0'])
    kpt1 = kpt1.clone().detach()[:, [1, 0]].to(data['ft0'])
    _, h, w = data['s0'].shape
    scale = torch.tensor([w, h]).to(kpt0)
    desc0 = sample_descriptors_fix_sampling(kpt0, data['ft0'], scale)
    desc1 = sample_descriptors_fix_sampling(kpt1, data['ft1'], scale)
    mlp = get_mlp_model(16).to("cuda")
    en = time.time()
    st = time.time()
    tmp = {}
    tmp["keypoints0"] = kpt0.to(device).unsqueeze(0).float()
    tmp["keypoints1"] = kpt1.to(device).unsqueeze(0).float()
    tmp["descriptors0"] = mlp.decode(desc0.to(device)).float()
    tmp["descriptors1"] = mlp.decode(desc1.to(device)).float()
    tmp["image_size"] = data['s0'].shape[1:]
    pred = matcher(tmp)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[0][valid]].cpu()
    data['mkpt0'] = m0
    data['mkpt1'] = m1
    data['kpt0'] = kpt0
    data['kpt1'] = kpt1


transfrom = transforms.ToTensor()
def encoder_img_match(data: dict, encoder, matcher) -> Tuple[torch.Tensor, torch.Tensor]:
    d0 = {}
    d1 = {}
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
    if p0['keypoints'].shape[1]==0 or p1['keypoints'].shape[1]==0:
        data['mkpt0'] = None
        data['mkpt1'] = None
        data['kpt0'] = tmp['keypoints0'][0].cpu()
        data['kpt1'] = tmp['keypoints0'][0].cpu()
        return
    pred = matcher(tmp)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = tmp["keypoints0"][0][valid].cpu(), tmp["keypoints1"][0][m0[0][valid]].cpu()
    kpt0, kpt1 = tmp['keypoints0'][0].cpu(), tmp['keypoints1'][0].cpu()
    data['mkpt0'] = m0
    data['mkpt1'] = m1
    data['kpt0'] = kpt0
    data['kpt1'] = kpt1


def semi_img_match(data: dict, matcher) -> Tuple[torch.Tensor, torch.Tensor]:
    tmp = {}
    tmp['image0'] = data['img0']
    tmp['image1'] = data['img1']
    with torch.no_grad():   
      matcher(tmp, online_resize=True)
      corr0, corr1 = tmp['mkpts0_f'].cpu(), tmp['mkpts1_f'].cpu()
    data['mkpt0'] = corr0
    data['mkpt1'] = corr1


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


def img_match_test(data: dict, encoder, matcher, mlp) -> Tuple[torch.Tensor, torch.Tensor]:
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
    tmp["descriptors0"] = mlp.back(p0['descriptors'])
    tmp["descriptors1"] = mlp.back(p1['descriptors'])
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


def save_matchimg(data, path):
    all_images, all_keypoints, all_matches = [], [], []
    all_images.append([data["img0"].detach().cpu().numpy(), data["img1"].detach().cpu().numpy()])
    all_matches.append((data['mkpt0'].detach().cpu(), data['mkpt1'].detach().cpu()))
    fig, axes = plot_image_grid(all_images, return_fig=True, set_lim=True)
    if data.get('kpt0') is not None:
        all_keypoints.append([data['kpt0'].cpu(), data['kpt1'].cpu()])
        plot_keypoints(all_keypoints[0], axes=axes[0], colors="royalblue")
    plot_matches(*all_matches[0], color=None, axes=axes[0], alpha=0.5, line_width=1.0, point_size=0.0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def match_data(data, matcher, img0, img1):
    pred = matcher(data)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m0, m1 = data["keypoints0"][0][valid].cpu(), data["keypoints1"][0][m0[0][valid]].cpu()
    result = {}
    result['img0'] = img0
    result['img1'] = img1
    result['mkpt0'] = m0
    result['mkpt1'] = m1
    result['kpt0'] = data["keypoints0"].squeeze(0).cpu()
    result['kpt1'] = data["keypoints1"].squeeze(0).cpu()
    return result


def save_matchimg_th(data, path, threshold=5e-6):
    all_images, all_keypoints, all_matches = [], [], []
    all_images.append([data["img0"].detach().cpu().numpy(), data["img1"].detach().cpu().numpy()])
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
                  captions=[f"Matches: {match_num}",
                            f"Precision({threshold:.1e})({precision:.1f}%): {epi_good.sum()}/{len(epi_good)}",
                            f"R_err={R_err:.2f}, t_err={t_err:.2f}"])
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
