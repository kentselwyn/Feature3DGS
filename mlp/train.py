import os
import time
import torch
import logging
import torch.nn as nn
from .utils.tools import set_seed
from .utils.modules import get_mlp
from .utils.tensor import batch_to_device
from .utils.image import ImagePreprocessor, load_image
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from argparse import ArgumentParser
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from encoders.superpoint.superpoint import SuperPoint


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lr_schedule(epoch):
    start = lr_schedule_epoch
    if epoch < start:
        return 1.0
    else:
        factor = 0.9 ** ((epoch - start) // 500)
        return factor


def manage_checkpoints(save_dir, max_checkpoints=10):
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoint_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    if len(checkpoint_files) > max_checkpoints:
        for old_ckpt in checkpoint_files[:-max_checkpoints]:
            old_ckpt_path = os.path.join(save_dir, old_ckpt)
            os.remove(old_ckpt_path)


class feature_set(Dataset):
    def __init__(self, args, encoder) -> None:
        train_path = Path(args.path)/"train/rgb"
        test_path = Path(args.path)/"test/rgb"
        train_images = list(train_path.glob("**/" + "*.png"))
        test_images = list(test_path.glob("**/" + "*.png"))
        all_images = train_images + test_images
        preprocessor = ImagePreprocessor({})
        self.features = []
        with torch.no_grad():
            for idx, img_p in enumerate(all_images):
                img = load_image(img_p)
                data = {}
                data["image"] = img.unsqueeze(0).to(device)
                pred = encoder(data)
                feat = pred["descriptors"].squeeze(0).cpu()
                self.features.append(feat)
        logger.info(f"Loaded images!")
    def __getitem__(self, idx):
        desc = self.features[idx]
        return desc
    def __len__(self):
        return len(self.features)


def main(args, conf):
    encoder =  SuperPoint(conf).eval().to(device)
    # Initialize dataset and dataloader
    dataset = feature_set(args, encoder)
    loader = DataLoader(dataset, batch_size=args.batch_size , 
                             shuffle=True, num_workers=args.num_workers)
    # Model setup
    model = get_mlp(args.dim)
    model.to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params , lr=args.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    best_vloss = float("inf")
    l2_loss = nn.MSELoss()
    # Logging
    writer = SummaryWriter(f"{args.out_path}/runs/{timestamp}")
    logger.info("Training started.")
    for epoch in range(args.epochs):
        logger.info(f"EPOCH {epoch + 1}:")
        start_time = time.time()
        model.train(True)
        total_loss = 0
        
        for index, data in enumerate(loader):
            data = batch_to_device(data, device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data)
            loss = l2_loss(pred, data)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if index % 99 == 0:
                total_iteration = len(loader) * epoch + index
                writer.add_scalar("train/mse", loss.item(), total_iteration)
        total_loss /= len(loader)
        logger.info(f"[E {epoch} | iter {index}] loss {total_loss:.6f}")

        epoch_time = time.time() - start_time
        logger.info(f'Time taken for EPOCH {epoch + 1}: {epoch_time:.2f} seconds')
        if total_loss < best_vloss:
            logger.info(f'New checkpoint! total_loss: {total_loss:.6f}')
            best_vloss = total_loss
            checkpoint_path = f'{args.out_path}/epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            manage_checkpoints(args.out_path, max_checkpoints=10)


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--lr_schedule_epoch", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_kpts", type=int, default=1024)
    parser.add_argument("--sp_th", type=float, default=0.0)
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.out_path+"/log.txt", mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    lr_schedule_epoch = args.lr_schedule_epoch
    model_conf = {
        "max_num_keypoints": args.num_kpts,
        "detection_threshold": args.sp_th,
        "force_num_keypoints": True,
    }
    model_conf = OmegaConf.create(model_conf)
    main(args, model_conf)
