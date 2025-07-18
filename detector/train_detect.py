import os
import gc
import time
import torch
import logging
from torch import nn
from scene import Scene
from datetime import datetime
from argparse import ArgumentParser
from encoders.superpoint.superpoint import SuperPoint
from scene.gaussian.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from Feature3DGS.gaussian_renderer.__init__edit import render
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from mlp.train import manage_checkpoints
from utils.loc.loc_utils import choose_th
from .utils import combined_loss, weighted_l2, peakness_loss, normalize_to_01, \
                    remove_neg_and_normalize
import torchvision.utils as vutils
from utils.match.match_img import find_small_circle_centers, sample_descriptors_fix_sampling
from utils.plot import plot_points


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



# regression distribution, scoremap may be negative
class ScoreMap_set(Dataset):
    def __init__(self, scene:Scene, split="train") -> None:
        self.images = []
        self.scoremaps = []
        if split=="train":
            views = scene.getTrainCameras()
        elif split=="test":
            views = scene.getTestCameras()
        bg_color = [1,1,1] if model_param.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(f"start build dataset {split}")
        for idx, view in enumerate(views):
            print(idx)
            pkg = render(view, scene.gaussians, Pipe_param.extract(args), background)
            image = view.original_image.cpu().clone().pin_memory()
            # .detach().cpu().clone().pin_memory()
            scoremap = pkg['score_map']
            _, h, w = scoremap.shape
            th = choose_th(scoremap, args.hist)
            kpts = find_small_circle_centers(scoremap, threshold=th, kernel_size=15).clone().detach()[:, [1, 0]]
            score = torch.zeros((1, h, w), dtype=torch.float32).cuda()
            plot_points(score, kpts)
            self.images.append(image)
            if args.sparse:
                score = score.detach().cpu().clone().pin_memory()
                self.scoremaps.append(score)
            else:
                scoremap = scoremap.detach().cpu().clone().pin_memory()
                self.scoremaps.append(scoremap)
        views.clear()
        logger.info(f"Loaded images!")
        del views
        gc.collect()
    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.scoremaps[idx])
    def __len__(self):
        return len(self.scoremaps)


def lr_schedule(epoch):
    start = lr_schedule_epoch
    if epoch < start:
        return 1.0
    else:
        factor = 0.9 ** ((epoch - start) // 500)
        return factor


def ensure_three_channels(t):
    if t.ndim == 2:
        t = t.unsqueeze(0)  # [H, W] -> [1, H, W]
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
    return t


def main(scene: Scene, encoder):
    if args.model_type==1:
        from .models import RefinedScoreNet as Net
    elif args.model_type==2:
        from .models import RefinedScoreNet2 as Net
    elif args.model_type==3:
        from .models import RefinedScoreNet3 as Net
    model = Net(encoder, args.sptrain).cuda()
    traindataset = ScoreMap_set(scene, split="train")
    testdataset = ScoreMap_set(scene, split="test")
    del scene
    torch.cuda.empty_cache()
    train_loader = DataLoader(traindataset, batch_size=4 , 
                        shuffle=True, num_workers=0, 
                        pin_memory=True)
    test_loader = DataLoader(testdataset, batch_size=1 , 
                        shuffle=False, num_workers=0, 
                        pin_memory=True)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params , lr=args.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    writer = SummaryWriter(f"{args.out_path}/runs/{timestamp}")
    best_total_loss = float("inf")
    if args.loss_type=="L2":
        loss_fn = nn.MSELoss()
    elif args.loss_type=="L1":
        loss_fn = nn.L1Loss()
    elif args.loss_type=="weightedL2Sharp":
        loss_fn = combined_loss
    elif args.loss_type=="weightedL2":
        loss_fn = weighted_l2

    logger.info("Training started.")

    for epoch in range(args.epochs):
        logger.info(f"EPOCH {epoch}:")
        start_time = time.time()
        model.train()
        total_loss = 0
        for idx, data in enumerate(train_loader):
            img, scoremap = data
            img = img.to(device='cuda', non_blocking=True)
            scoremap = scoremap.to(device='cuda', non_blocking=True)
            optimizer.zero_grad()
            pred = model(img)
            scoremap = remove_neg_and_normalize(scoremap)
            # loss_peak = peakness_loss(norm_pred)
            loss = loss_fn(pred, scoremap)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (idx+1) % 250 == 0:
                total_iteration = len(train_loader) * epoch + idx
                writer.add_scalar("train/loss", loss.item(), total_iteration)
                #
                pred_heatmap_img = pred[0].detach().cpu()
                pred_heatmap_img = (pred_heatmap_img - pred_heatmap_img.min()) / (pred_heatmap_img.max() - pred_heatmap_img.min() + 1e-6)
                #
                gt_heatmap_img = scoremap[0].detach().cpu()
                gt_heatmap_img = (gt_heatmap_img - gt_heatmap_img.min()) / (gt_heatmap_img.max() - gt_heatmap_img.min() + 1e-6)
                #
                gt_img = img[0].detach().cpu()
                gt_img = ensure_three_channels(gt_img)
                pred_heatmap_img = ensure_three_channels(pred_heatmap_img)
                gt_heatmap_img   = ensure_three_channels(gt_heatmap_img)
                grid = vutils.make_grid([gt_img, 
                                         pred_heatmap_img, 
                                         gt_heatmap_img], nrow=3, padding=2)
                writer.add_image(f"train/composite/{idx}", grid, global_step=epoch)
        total_loss /= len(train_loader)
        logger.info(f"[E {epoch} | iter {idx}] loss {total_loss:.6f}")
        writer.add_scalar("train/loss_total", total_loss, total_iteration)
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                img, scoremap = data
                img = img.to(device='cuda', non_blocking=True)
                scoremap = scoremap.to(device='cuda', non_blocking=True)
                pred = model(img)
                pred_heatmap_img = pred[0].detach().cpu()
                pred_heatmap_img = (pred_heatmap_img - pred_heatmap_img.min()) / (pred_heatmap_img.max() - pred_heatmap_img.min() + 1e-6)
                #
                gt_heatmap_img = scoremap[0].detach().cpu()
                gt_heatmap_img = (gt_heatmap_img - gt_heatmap_img.min()) / (gt_heatmap_img.max() - gt_heatmap_img.min() + 1e-6)
                #
                gt_img = img[0].detach().cpu()
                gt_img = ensure_three_channels(gt_img)
                pred_heatmap_img = ensure_three_channels(pred_heatmap_img)
                gt_heatmap_img   = ensure_three_channels(gt_heatmap_img)
                grid = vutils.make_grid([gt_img, 
                                         pred_heatmap_img, 
                                         gt_heatmap_img], nrow=3, padding=2)
                writer.add_image(f"test/composite/{idx}", grid, global_step=epoch)

        logger.info(f'Time taken for EPOCH {epoch}: {time.time() - start_time:.2f} seconds')
        if total_loss < best_total_loss:
            logger.info(f'New checkpoint! total_loss: {total_loss:.6f}')
            best_total_loss = total_loss
            checkpoint_path = f'{args.out_path}/epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            manage_checkpoints(args.out_path, max_checkpoints=5)
    

# ( bash z_scripts/tmp/train_detect.sh )
if __name__=="__main__":
    parser = ArgumentParser(description="Testing script parameters")
    Model_param = ModelParams(parser, sentinel=True)
    Pipe_param = PipelineParams(parser)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr_schedule_epoch", type=int, default=100)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss_type", type=str, default="L2")
    parser.add_argument("--hist", default=0.95, type=float)
    parser.add_argument("--model_type", type=int, default=1)
    parser.add_argument("--sptrain", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    args = get_combined_args(parser)
    model_param = Model_param.extract(args)
    gaussians = GaussianModel(model_param.sh_degree)
    scene = Scene(model_param, gaussians, load_iteration=args.iteration, shuffle=False,
                    load_test_cams=True, load_feature=False, view_num=10, test_only_view_num=True)
    os.makedirs(args.out_path, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.out_path+"/log.txt", mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    lr_schedule_epoch = args.lr_schedule_epoch
    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": args.num_kpts,
        "detection_threshold": float(args.detect_th),
    }
    render_encoder = SuperPoint(conf).cuda().train()
    main(scene, render_encoder)
    