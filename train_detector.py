import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
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



class ImageScoreDataset(Dataset):
    def __init__(self, images, scores, transform=None):
        assert len(images) == len(scores), "圖片和 score 數量必須相同"
        self.images = images
        self.scores = scores
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        score = self.scores[idx]
        if self.transform:
            image = self.transform(image)
        return image, score

def normalize_score_map(score):
    min_val = score.amin(dim=(2, 3), keepdim=True)
    max_val = score.amax(dim=(2, 3), keepdim=True)
    # breakpoint()
    return (score - min_val) / (max_val - min_val + 1e-8)


# def score_map_bce_loss(score_map_logits, gt_map):
#     """
#     score_map_logits: raw output from detector, shape [B, 1, H, W]
#     gt_map: ground truth map, shape [B, 1, H, W]
#     """
#     # 將 raw logits 通過 sigmoid → [0, 1]
#     pred = torch.sigmoid(score_map_logits)

#     # BCE loss (預設 mean)
#     loss = F.binary_cross_entropy(pred, gt_map.float())
#     return loss


def score_map_bce_loss2(score_map, gt_map):
    score_map = score_map.flatten()
    gt_map = gt_map.flatten()
    loss = F.binary_cross_entropy(score_map, gt_map.float())

    return loss




def evaluate_detector(
    detector,
    pipe_param,
    feature_extractor,
    gaussians,
    scene,
    masks,
    tb_writer,
    iteration,
):
    viewpoint_stack = scene.getTestCameras().copy()
    conf = {
            "sparse_outputs": True,
            "dense_outputs": True,
            "max_num_keypoints": 1024,
            "detection_threshold": 0.01,
    }
    # feature_extractor = SuperPoint(conf).to("cuda").eval()
    bg_color = [0, 0, 0]
    all_features = []
    all_scores = []
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    loss_sum = 0.0
    for idx, viewpoint_cam in enumerate(viewpoint_stack):
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

        loss = score_map_bce_loss2(heat_map, gt_map)
        loss_sum += loss.item()

        if tb_writer and idx < 10:
            rendering = render_pkg["render"]
            heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())
            tb_writer.add_images(f"detector_vis_test/gt_map_{idx}", gt_map, iteration,)
            tb_writer.add_images(f"detector_vis_test/heat_map_{idx}", heat_map,iteration,)
            tb_writer.add_images(f"detector_vis_test/render_{idx}", rendering[None],iteration,)
    loss_sum /= len(viewpoint_stack)
    print(f"\n[ITER {iteration}] Evaluating detector: test loss {loss_sum}")
    if tb_writer:
        tb_writer.add_scalar(f"detector_loss_patches/test_loss", loss_sum, iteration,)


def training_detector(
    args,
    pipe_param,
    gaussians,
    scene: Scene,
    masks,
    testing_iterations,
    saving_iterations,
    tb_writer,
    train_iteration=30000,
    detector_folder="",
):
    viewpoint_stack = scene.getTrainCameras().copy()
    conf = {
            "sparse_outputs": True,
            "dense_outputs": True,
            "max_num_keypoints": 1024,
            "detection_threshold": 0.01,
    }
    feature_extractor = SuperPoint(conf).to("cuda").eval()
    bg_color = [0, 0, 0]
    all_features = []
    all_scores = []
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    detector = KpDetector(256).cuda().train()
    
    
    optimizer = torch.optim.AdamW(detector.parameters(), lr=0.001)
    grad_accum = 8
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_iteration // grad_accum, eta_min=0.0005
    )
    save_ckpt_path = os.path.join(scene.model_path, detector_folder, "ckpt")
    os.makedirs(save_ckpt_path, exist_ok=True)
    idx = 0
    while len(viewpoint_stack) > 0:
        # print(idx)
        idx+=1
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        render_pkg = render(viewpoint_cam, gaussians, pipe_param, background)
        gt_image = viewpoint_cam.original_image.cuda()
        data = {"image":gt_image[None]}
        gt_feature_map = feature_extractor(data)["dense_descriptors"].squeeze(0)

        all_features.append(gt_feature_map.detach().cpu())
        score_map = render_pkg["score_map"].detach()
        kpts = extract_kpt(score_map, threshold=0.3)

        gt_score_map = fast_render_gaussian_kpt_map(kpts, shape=score_map.shape[1:], sigma=1.0, device="cuda")
        img = gt_score_map[0].detach().cpu().numpy()
        all_scores.append(gt_score_map)

    dataset = ImageScoreDataset(all_features, all_scores)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    iteration = 0  # 全域計數器
    progress_bar = tqdm(total=train_iteration, desc="Scene-Specific Detector")
    data_iter = iter(dataloader)
    first_iter = 1
    for iteration in range(first_iter, train_iteration + 1):
        try:
            features, scores = next(data_iter)
        except StopIteration:
            # 如果跑完一輪，重新打亂並重啟迭代器
            data_iter = iter(dataloader)
            features, scores = next(data_iter)

        features = features.cuda()
        features = F.interpolate(
            features,
            size=(gt_image.shape[1], gt_image.shape[2]),
            mode="bilinear",
            align_corners=False,
        )
        scores = scores.cuda()
        features = F.normalize(features, p=2, dim=1)
        gt_map = normalize_score_map(scores)
        heat_map = detector(features)
        loss = score_map_bce_loss2(heat_map, gt_map)
        loss.backward()
        if iteration % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            lr_scheduler.step()

        with torch.no_grad():
            # Progress bar
            loss_val = loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss_val:.{7}f}",})
                progress_bar.update(10)
            if iteration == train_iteration:
                progress_bar.close()
            if tb_writer:
                tb_writer.add_scalar("detector_loss_patches/training_loss", loss_val, iteration)
                tb_writer.add_scalar("detector_loss_patches/lr", optimizer.param_groups[0]["lr"], iteration,)
        if iteration in testing_iterations:
            print("\n[ITER {}] Evaluating detector".format(iteration))
            detector.eval()
            evaluate_detector(detector, pipe_param, feature_extractor, gaussians, scene, masks, tb_writer,
                            iteration,)
            detector.train()
        if iteration in saving_iterations:
            print("\n[ITER {}] Saving detector".format(iteration))
            torch.save(detector.state_dict(), save_ckpt_path + f"/{iteration}_detector.pth")


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

    args = get_combined_args(parser)
    dataset = lp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, 
                  test_feature_load=False, load_semantic_feature=False, load_test_cams=True)

    masks = None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = SummaryWriter(
        os.path.join(dataset.model_path, args.detector_folder, "runs", timestamp)
    )
    training_detector(
        args,
        Pipe_param.extract(args),
        gaussians,
        scene,
        masks,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        tb_writer=tb_writer,
        train_iteration=30000,
        detector_folder=args.detector_folder,
    )
