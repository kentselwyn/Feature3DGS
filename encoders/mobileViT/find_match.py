from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation
import torch
import cv2
from PIL import Image
import numpy as np
import requests
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from encoders.superpoint.superpoint import SuperPoint
from matchers.MNN import NearestNeighborMatcher
from utils.match.viz2d import plot_matches, plot_image_grid, plot_keypoints
from mlp.mlp import get_mlp_vit


def sample_descriptors_fix_sampling(kpt, desc, scale):
    b, c, _, _ = desc.shape
    kpt = kpt / scale
    kpt = kpt*2 - 1
    kpt = kpt.float()
    desc = desc.float() # add batch dim
    desc = torch.nn.functional.grid_sample(desc, kpt.view(1, 1, -1, 2), 
                                           mode="bilinear", align_corners=False)
    desc = desc.reshape(1, c, -1).transpose(-1,-2)
    return desc


# python find_match.py
if __name__=="__main__":
    img_p0 = "/home/koki/code/cc/feature_3dgs_2_copy/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs/test/rgb/seq-01-frame-000135.color.png"
    img_p1 = "/home/koki/code/cc/feature_3dgs_2_copy/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs/test/rgb/seq-01-frame-000137.color.png"
    
    bgr_image = cv2.imread(img_p0)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img0 = Image.fromarray(rgb_image)

    bgr_image = cv2.imread(img_p1)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img1 = Image.fromarray(rgb_image)

    processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
    model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")
    model.eval().cuda()

    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": 1024,
        "detection_threshold": 0.01,
    }
    encoder = SuperPoint(conf).cuda().eval()
    matcher = NearestNeighborMatcher({}).cuda().eval()

    input0 = processor(images=img0, return_tensors="pt")
    input1 = processor(images=img1, return_tensors="pt")
    input0["pixel_values"] = input0["pixel_values"].cuda()
    input1["pixel_values"] = input1["pixel_values"].cuda()
    data0 = {}
    data1 = {}
    data0["image"] = input0["pixel_values"]
    data1["image"] = input1["pixel_values"]

    pred0 = encoder(data0)
    pred1 = encoder(data1)

    kpt0 = pred0["keypoints"]
    kpt1 = pred1["keypoints"]

    with torch.no_grad():
        feat0 = model.base_model(**input0).last_hidden_state
        feat1 = model.base_model(**input1).last_hidden_state

    b,c,h,w = data0["image"].shape
    scale = torch.tensor([w, h]).to(kpt0)

    desc0 = sample_descriptors_fix_sampling(kpt0, feat0, scale)
    desc1 = sample_descriptors_fix_sampling(kpt1, feat1, scale)

    data = {}
    data["desc0"] = desc0
    data["desc1"] = desc1

    pred = matcher(data)
    m0 = pred['m0']
    valid = (m0[0] > -1)
    m_kpts0, m_kpts1 = kpt0[0][valid].cpu().numpy(), kpt1[0][m0[0][valid]].cpu().numpy()
    
    img0 = data0["image"][0].permute(1, 2, 0).cpu().numpy()
    img1 = data1["image"][0].permute(1, 2, 0).cpu().numpy()
    all_images, all_keypoints, all_matches = [], [], []
    all_images.append([img0, img1])
    all_keypoints.append([kpt0[0].to("cpu"), kpt1[0].to("cpu")])
    all_matches.append((m_kpts0, m_kpts1))

    fig, axes = plot_image_grid(all_images, return_fig=True, set_lim=True)
    plot_keypoints(all_keypoints[0], axes=axes[0], colors="royalblue")
    plot_matches(*all_matches[0], color=None, axes=axes[0], alpha=0.5, line_width=1.0, point_size=0.0)
    
    n0 = Path(img_p0).stem
    n1 = Path(img_p1).stem
    fig.savefig(f"{n0}_{n1}.png")
    
    # breakpoint()
