import os
import time
import torch
import h5py
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
import torch.nn as nn
from argparse import ArgumentParser
from .modules import get_mlp
from .tools import set_seed
from .tensor import batch_to_device
from .mlp_utils import do_evaluation3
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, random_split, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_scene(data_list:list, scene_path:str, conf):
    data_name = Path(scene_path).parent.name
    scene_name = Path(scene_path).name
    if scene_name.startswith("Cambridge") or scene_name.startswith("pgt_7scenes"):
        print(scene_name)
        h_path = f"{scene_path}/desc_data/r640_SP-k1024-nms4-{data_name}_{scene_name}"+\
                    f"-aug{conf.augmentation}_setlen{conf.multiple}.h5"
    else:
        return
    with h5py.File(h_path, 'r') as hfile:
        root_group = hfile['home']
        print('start loading.....')
        def gather_data(group):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    gather_data(item)
                elif isinstance(item, h5py.Dataset) and 'descriptors' in key:
                    descriptors = group['descriptors'][:]
                    data_list.append(descriptors)
        gather_data(root_group)


class MLPDataset(Dataset):
    def __init__(self, all_path, data_names, conf):
        self.data = []
        self.keypoints = []
        for data_name in os.listdir(all_path):
            if data_name in data_names:
                data_path = f"{all_path}/{data_name}"
                for scene_name in os.listdir(data_path):
                    scene_path = f"{data_path}/{scene_name}"
                    load_scene(self.data, scene_path, conf)
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset = MLPDataset(conf.all_path, conf.data_names, conf)
    set_seed(conf.train.seed)

    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    trainloader = DataLoader(train_dataset, batch_size=conf.train.batch_size , shuffle=True, num_workers=conf.train.num_workers)
    valloader = DataLoader(val_dataset, batch_size=conf.train.batch_size, shuffle=False, num_workers=conf.train.num_workers)

    out_path = f"{conf.mlp_folder_path}/" + \
                f"type:{MODEL}_time:{timestamp}_dim{conf.dim}_batch{conf.train.batch_size}_" + \
                f"lr{conf.train.lr}_epoch{conf.train.epochs}_-aug{conf.augmentation}_setlen{conf.multiple}"
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


# nohup python -u -m mlp.train_many --dim 4 > /home/koki/code/cc/feature_3dgs_2/log_lg_4_mlp.txt 2>&1 &
# nohup python -u -m mlp.train_many --dim 8 > /home/koki/code/cc/feature_3dgs_2/log_lg_8_mlp.txt 2>&1 &
# nohup python -u -m mlp.train_many --dim 16 > /home/koki/code/cc/feature_3dgs_2/log_lg_16_mlp.txt 2>&1 &
if __name__=="__main__":
    data_names = ["Cambridge", "7_scenes"]
    MODEL = "SP"
    all_path = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc"
    conf = {
        "dim": None,
        "mlp_folder_path": f"{all_path}/mlpckpt",
        "data_names": data_names,
        "all_path": f"{all_path}",
        "augmentation": "lg",
        "multiple": 3,
        "load_ckpt": None,
        "train":{
            "epochs": 5000,
            "start_epoch": 2500,
            "lr": 8e-4,
            "seed": 0,
            "batch_size": 64,
            "num_workers": 0,
        }
    }
    conf = OmegaConf.create(conf)
    parser = ArgumentParser()
    parser.add_argument(
        "--dim",
        type=int,
    )
    args = parser.parse_args()
    conf.dim = args.dim
    main(conf=conf)
