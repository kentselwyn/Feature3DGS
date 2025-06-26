import os
import time
import torch
import torch.nn as nn
from .utils.tools import set_seed
from .utils.modules import get_mlp
from .utils.tensor import batch_to_device
from .utils.mlp_utils import do_evaluation3
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
                print(idx, img_p)
                img = load_image(img_p)
                data = {}
                data["image"] = img.unsqueeze(0).to(device)
                pred = encoder(data)
                feat = pred["descriptors"].squeeze(0).cpu()
                self.features.append(feat)
    def __getitem__(self, idx):
        desc = self.features[idx]
        return desc
    def __len__(self):
        return len(self.features)


def main(args, conf):
    encoder =  SuperPoint(conf).eval().to(device)
    dataset = feature_set(args, encoder)
    model = get_mlp(args.dim)
    model.to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params , lr=args.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    best_vloss = 1_000_000.
    l2_loss = nn.MSELoss()
    loader = DataLoader(dataset, batch_size=args.batch_size , 
                             shuffle=True, num_workers=args.num_workers)
    os.makedirs(args.out_path, exist_ok=True)
    writer = SummaryWriter(f"{args.out_path}/runs/{timestamp}")
    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch + 1))
        start_time = time.time()
        model.train(True)
        for index, data in enumerate(loader):
            data = batch_to_device(data, device, non_blocking=True)
            breakpoint()


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
    lr_schedule_epoch = args.lr_schedule_epoch
    model_conf = {
        "max_num_keypoints": args.num_kpts,
        "detection_threshold": args.sp_th,
        "force_num_keypoints": True,
    }
    model_conf = OmegaConf.create(model_conf)
    main(args, model_conf)
