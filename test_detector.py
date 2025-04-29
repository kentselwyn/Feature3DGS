import os
import sys
import uuid
import torchvision
from argparse import ArgumentParser, Namespace
import random
from random import choice
from PIL import Image
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from arguments import ModelParams, OptimizationParams, get_combined_args, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False   
from encoders.superpoint.superpoint import SuperPoint
from gaussian_renderer import render
from scene.kpdetector import KpDetector
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils.match_img import extract_kpt, render_gaussian_kpt_map, fast_render_gaussian_kpt_map
import numpy as np
from datetime import datetime
from render import feature_visualize_saving
from train_detector import normalize_score_map

def save_grayscale_map(tensor, filename):
    img = tensor[0, 0].detach().cpu().numpy()  # [H, W]
    img = (img / img.max() * 255.0).astype("uint8")
    Image.fromarray(img).save(filename)

random.seed(1000)

conf = {
    "sparse_outputs": True,
    "dense_outputs": True,
    "max_num_keypoints": 1024,
    "detection_threshold": 0.01,
}


def save_detect_kpt_img(
    args,
    pipe_param,
    gaussians,
    scene: Scene,
    masks,
    testing_iterations,
    saving_iterations,
    iteration=30000,
    detector_folder="",
    num = 10,
):
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # breakpoint()
    viewpoint_stack = scene.getTrainCameras().copy()
    
    ckpt_path = os.path.join(scene.model_path, detector_folder, "ckpt", f"{iteration}_detector.pth")
    img_folder = os.path.join(scene.model_path, detector_folder, "imgs", "train")
    os.makedirs(img_folder, exist_ok=True)
    detector = KpDetector(256).cuda().eval()
    state_dict = torch.load(ckpt_path, map_location='cuda')  # 若在 CPU 上可改 'cpu'
    detector.load_state_dict(state_dict)
    
    feature_extractor = SuperPoint(conf).to("cuda").eval()
    length = len(viewpoint_stack)
    # step = length // num
    # lst = [i for i in range(0, length-1, step)]
    # breakpoint()
    # n = len(viewpoint_stack)
    for i in range(length):
        print(i)
        # idx = randint(0, len(viewpoint_stack) - 1)
        # viewpoint_cam = viewpoint_stack[idx]
        viewpoint_cam = viewpoint_stack[i]
        image_name = viewpoint_cam.image_name

        render_pkg = render(viewpoint_cam, gaussians, pipe_param, background)
        
        gt_image = viewpoint_cam.original_image.cuda()
        data = {"image":gt_image[None]}
        
        gt_feature_map = feature_extractor(data)["dense_descriptors"]
        gt_feature_map = F.interpolate(
            gt_feature_map,
            size=(gt_image.shape[1], gt_image.shape[2]),
            mode="bilinear",
            align_corners=False,
        )
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
        score_map = render_pkg["score_map"].detach()
        kpts = extract_kpt(score_map, threshold=0.3)
        gt_score_map = fast_render_gaussian_kpt_map(kpts, shape=score_map.shape[1:], sigma=1.0, device="cuda").unsqueeze(0)
        
        gt_map = normalize_score_map(gt_score_map)
        heat_map = detector(gt_feature_map)

        rendering = render_pkg["render"].detach()
        # gt_feature_map

        # img = rendering.detach().cpu().permute(1, 2, 0).numpy()
        img_fold = f"{img_folder}/{image_name}"
        os.makedirs(f"{img_fold}", exist_ok=True)
        torchvision.utils.save_image(render_pkg["render"], f"{img_fold}/rendering_rgb.png")
        # plt.imsave(f"{img_fold}/rendering_rgb.png", img)
        save_grayscale_map(gt_map, f"{img_fold}/gt_map.png")
        save_grayscale_map(heat_map, f"{img_fold}/heat_map.png")
        feature_map_vis = feature_visualize_saving(gt_feature_map.detach().cuda().squeeze(0))
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(
            os.path.join(f"{img_fold}/gt_feature_map.png"))
        # breakpoint()




if __name__=="__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--detector_folder", type=str, default="detector")
    parser.add_argument("--landmark_num", type=int, default=16384)
    parser.add_argument("--landmark_k", type=int, default=32)
    parser.add_argument("--view_num", type=int, default=10)

    args = get_combined_args(parser)
    dataset = lp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=True,
                  test_feature_load=False, load_semantic_feature=False, load_test_cams=True,
                  view_num=args.view_num, load_feature=False)

    masks = None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    save_detect_kpt_img(
        args,
        Pipe_param.extract(args),
        gaussians,
        scene,
        masks,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        detector_folder=args.detector_folder,
    )
