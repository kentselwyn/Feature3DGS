import os
import h5py
import time
import torch
import torch.nn as nn
from .tools import set_seed
from .modules import get_mlp
from datetime import datetime
from omegaconf import OmegaConf
from argparse import ArgumentParser
from .tensor import batch_to_device
from .mlp_utils import do_evaluation3
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, random_split, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

class MLPDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        self.data = []
        self.keypoints = []
        with h5py.File(self.h5_file_path, 'r') as hfile:
            root_group = hfile['home']
            print('start loading.....')
            def gather_data(group):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        gather_data(item)
                    elif isinstance(item, h5py.Dataset) and 'descriptors' in key:
                        desc = group['descriptors'][:]
                        self.data.append(desc)
                    elif isinstance(item, h5py.Dataset) and 'dense_descriptors' in key:
                        desc = group['dense_descriptors'][:]
                        self.data.append(desc)
            gather_data(root_group)
        print('loaded!')

    def __getitem__(self, idx):
        desc = torch.tensor(self.data[idx], dtype=torch.float32)
        return desc
    
    def __len__(self):
        return len(self.data)


def lr_schedule(epoch):
    start = conf['train']['start_epoch']
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


def main(conf):
    conf = OmegaConf.create(conf)
    model = get_mlp(conf.dim)
    model.to(device)

    if conf.load_ckpt is not None:
        ckpt = torch.load(conf.load_ckpt)
        model.load_state_dict(ckpt)
        print(f'{conf.load_ckpt} ckpt loaded')

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params , lr=conf.train.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    best_vloss = 1_000_000.
    l2_loss = nn.MSELoss()

    dataset = MLPDataset(conf.fea_path)
    set_seed(conf.train.seed)

    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    trainloader = DataLoader(train_dataset, batch_size=conf.train.batch_size , 
                             shuffle=True, num_workers=conf.train.num_workers)
    valloader = DataLoader(val_dataset, batch_size=conf.train.batch_size, 
                           shuffle=False, num_workers=conf.train.num_workers)

    out_path = f"{conf.folder_path}/{conf.out_name}"
    os.makedirs(out_path, exist_ok=True)

    writer = SummaryWriter(f"{out_path}/runs/{timestamp}")

    for epoch in range(conf.train.epochs):
        print('EPOCH {}:'.format(epoch + 1))
        start_time = time.time()
        model.train(True)
        for index, data in enumerate(trainloader):
            data = batch_to_device(data, device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data)
            loss = l2_loss(pred, data)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if index % 99 == 0:
                total_iteration = len(trainloader) * epoch + index
                writer.add_scalar("train/" + "mse", loss.item(), total_iteration)

        print( "[E {} | iter {}] loss {}".format(epoch, index, loss.item()))
        val_loss = do_evaluation3(model, valloader, device)
        print(f'[Validation]', val_loss.item())
        writer.add_scalar("val/" + 'mse', val_loss.item(), total_iteration)

        end_time = time.time()
        epoch_time = end_time - start_time
        print('Time taken for one EPOCH {}: {:.2f} seconds'.format(epoch + 1, epoch_time))

        if val_loss < best_vloss:
            print(f'new ckpt get! val_loss:{val_loss}\n')
            best_vloss = val_loss
            checkpoint_path = f'{out_path}/epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            manage_checkpoints(out_path, max_checkpoints=10)



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--scene_name", type=str)
    parser.add_argument("--desc_name", type=str)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--start_epoch", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    all_path = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/{args.data_name}/{args.scene_name}"
    conf = {
        "dim": args.dim,
        "folder_path": f"{all_path}/mlpckpt",
        "fea_path": f"{all_path}/desc_data/{args.desc_name}.h5",
        "out_name": f"type:SP_time:{timestamp}_dim{args.dim}_batch{args.batch_size}_lr{args.lr}_epoch{args.epochs}_desc{args.desc_name}",
        "load_ckpt": None,
        "train":{
            "epochs": args.epochs,
            "start_epoch": args.start_epoch,
            "lr": args.lr,
            "seed": 0,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
    }
    conf = OmegaConf.create(conf)
    main(conf=conf)
