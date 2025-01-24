import os
import time
import torch
import h5py
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
import torch.nn as nn
from argparse import ArgumentParser
from torch.utils.data import Dataset, random_split, DataLoader
from .modules import get_mlp, get_mlp_128
from .tools import set_seed
from .tensor import batch_to_device
from .mlp_utils import do_evaluation3
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_h5_data(file_path):
    with h5py.File(file_path, 'r') as hfile:
        desc = hfile['descriptors'][:]  # Read all data
        kpts = hfile['keypoints'][:]  # Read all labels

    return kpts, desc


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
                    # print(key)
                    if isinstance(item, h5py.Group):
                        gather_data(item)  # Recurse into subgroups
                    # and 'keypoints' in key and 
                    elif isinstance(item, h5py.Dataset) and 'descriptors' in key:
                        # Get the base path for this image group
                        base_path = group.name
                        
                        # Load descriptors and keypoints datasets
                        descriptors = group['descriptors'][:]
                        # keypoints = group['keypoints'][:]
                        
                        # Store them in memory
                        self.data.append(descriptors)
                        # print('k')
                        # self.keypoints.append(keypoints)
                        
            gather_data(root_group)
        
        print('loaded!')

    def __getitem__(self, idx):
        desc = torch.tensor(self.data[idx], dtype=torch.float32)
        return desc
    
    def __len__(self):
        return len(self.data)

# Cambridge
# 7_scenes
SCENE = "Cambridge"
# Cambridge_StMarysChurch
# pgt_7scenes_redkitchen
NAME = "Cambridge_StMarysChurch"
MODEL = "SP"
all_path = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/{SCENE}/{NAME}"



conf = {
    "dim": None,
    "folder_path": f"{all_path}/mlpckpt",
    "fea_path": f"{all_path}/desc_data/r1024_SP-k1024-nms4-{NAME}.h5",
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
def lr_schedule(epoch):
    start = conf['train']['start_epoch']
    if epoch < start:
        return 1.0  # Keep the learning rate the same
    else:
        factor = 0.9 ** ((epoch - start) // 500)  # Decrease every 500 epochs
        return factor



def manage_checkpoints(save_dir, max_checkpoints=10):
    # List all checkpoint files in the save directory
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    
    # Sort checkpoint files based on epoch number (assuming filenames are like 'epoch_X.pt')
    checkpoint_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

    # If more than 'max_checkpoints' checkpoints exist, remove the oldest ones
    if len(checkpoint_files) > max_checkpoints:
        for old_ckpt in checkpoint_files[:-max_checkpoints]:
            old_ckpt_path = os.path.join(save_dir, old_ckpt)
            os.remove(old_ckpt_path)
            # print(f"Removed old checkpoint: {old_ckpt_path}")




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
    dataset = MLPDataset(conf.fea_path)
    

    set_seed(conf.train.seed)

    dataset_size = len(dataset)
    train_size = int(0.98 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=conf.train.batch_size , shuffle=True, num_workers=conf.train.num_workers)
    valloader = DataLoader(val_dataset, batch_size=conf.train.batch_size, shuffle=False, num_workers=conf.train.num_workers)

    out_path = f"{conf.folder_path}/type:{MODEL}_time:{timestamp}_dim{conf.dim}_batch{conf.train.batch_size}_lr{conf.train.lr}_epoch{conf.train.epochs}"
    os.makedirs(out_path, exist_ok=True)

    writer = SummaryWriter(f"{out_path}/runs/{timestamp}")

    for epoch in range(conf.train.epochs):
        print('EPOCH {}:'.format(epoch + 1))
        start_time = time.time()
    
        # train_one_epoch(model, trainloader, epoch)
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
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }, checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path)

            manage_checkpoints(out_path, max_checkpoints=10)

    


def print_hdf5_structure(file_path):
    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r') as hfile:
        print(f"Structure of HDF5 file: {file_path}\n")
        # Recursively print the structure of the HDF5 file
        def recursive_print(name, item):
            if isinstance(item, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(item, h5py.Dataset):
                print(f"  Dataset: {name}, shape: {item.shape}, dtype: {item.dtype}")

        # Use hfile.visititems to traverse the structure
        hfile.visititems(recursive_print)



# nohup python -u -m mlp.train --dim 16 > /home/koki/code/cc/feature_3dgs_2/log_dim16_StMarysChurch.txt 2>&1 &
if __name__=="__main__":
    conf = OmegaConf.create(conf)
    parser = ArgumentParser()
    parser.add_argument(
        "--dim",
        type=int,
    )
    args = parser.parse_args()
    conf.dim = args.dim
    main(conf=conf)
    # print_hdf5_structure(conf.fea_path)




