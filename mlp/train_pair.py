import os
import torch
import time
import numpy as np
import h5py
import torch.nn.functional as F
from .modules import get_mlp
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
from mlp.mlp_utils import do_evaluation_pair
from torch.optim.lr_scheduler import ReduceLROnPlateau


global_descriptor_dict = {}




def downsample_npz(npz_name, output_name, keep_ratio=0.5,
                   all_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train/sparse/output_pairs"):
    data = np.load(f"{all_path}/{npz_name}", allow_pickle=True)
    n = len(data['img1'])
    k = int(n * keep_ratio)
    idxs = np.random.choice(n, k, replace=False)
    np.savez(
        f"{all_path}/{output_name}",
        img1 = data['img1'][idxs],
        xy1  = data['xy1'][idxs],
        img2 = data['img2'][idxs],
        xy2  = data['xy2'][idxs],
    )



def normalize_name(name):
    name = Path(name).with_suffix('').as_posix()     # 'seq-03/frame-000212.color'
    return name.replace('/', '-')


def load_dense_descriptors_to_mem(h5_file_path_list):
    """
    讀入多個 .h5 檔案中的 dense_descriptors 存到記憶體中
    回傳 dict: {normalized_name: descriptor_tensor (1, 256, H', W')}
    """
    feature_bank = {}
    for h5_path in h5_file_path_list:
        print(f"Loading: {h5_path}")
        with h5py.File(h5_path, "r") as h5:
            for key in h5.keys():
                if 'dense_descriptors' in h5[key]:
                    desc_np = h5[key]['dense_descriptors'][:]
                    desc_np = desc_np.astype(np.float32)
                    desc_np = np.expand_dims(desc_np, axis=0)  # (1, 256, H', W')
                    feature_bank[key] = desc_np
    print(f"Loaded {len(feature_bank)} descriptors into memory.")
    return feature_bank



class DensePairInMemDataset(Dataset):
    def __init__(self, npz_path, stride=8):
        self.data = np.load(npz_path, allow_pickle=True)
        self.img1 = self.data['img1']
        self.img2 = self.data['img2']
        self.xy1 = self.data['xy1']
        self.xy2 = self.data['xy2']
        self.label = self.data['label']
        # self.desc_dict = descriptor_dict
        self.stride = stride

    def _sample_descriptor(self, desc_map, xy):
        H, W = desc_map.shape[-2:]
        x = xy[0] / (W * self.stride - 1) * 2 - 1
        y = xy[1] / (H * self.stride - 1) * 2 - 1
        grid = torch.tensor([[[[x, y]]]], dtype=torch.float32, device=desc_map.device)  # (1,1,1,2)
        desc = F.grid_sample(desc_map, grid, align_corners=True).squeeze()  # (256,)
        return F.normalize(desc, dim=0)

    def __len__(self): return len(self.img1)

    def __getitem__(self, idx):
        name1 = normalize_name(self.img1[idx])
        name2 = normalize_name(self.img2[idx])
        xy1 = torch.tensor(self.xy1[idx], dtype=torch.float32)
        xy2 = torch.tensor(self.xy2[idx], dtype=torch.float32)

        desc_map1 = torch.from_numpy(global_descriptor_dict[name1])
        desc_map2 = torch.from_numpy(global_descriptor_dict[name2])

        f1 = self._sample_descriptor(desc_map1, xy1)
        f2 = self._sample_descriptor(desc_map2, xy2)
        label = torch.tensor(self.label[idx], dtype=torch.float32)

        return f1, f2, label



def contrastive_loss(z1, z2, label, margin=1.0):
    d = F.pairwise_distance(z1, z2, p=2)  # L2 距離
    loss = label * d.pow(2) + (1 - label) * F.relu(margin - d).pow(2)
    return loss.mean()



def l2_loss(z1, z2, label):
    # 只對正樣本計算 L2，相當於加強收斂性
    pos_mask = label == 1
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=z1.device)
    return F.mse_loss(z1[pos_mask], z2[pos_mask])




def manage_checkpoints(save_dir, max_checkpoints=10):
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoint_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    if len(checkpoint_files) > max_checkpoints:
        for old_ckpt in checkpoint_files[:-max_checkpoints]:
            old_ckpt_path = os.path.join(save_dir, old_ckpt)
            os.remove(old_ckpt_path)




def train(args):
    global global_descriptor_dict
    all_path = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/{args.scene}/train/sparse"
    h5_path=f"{all_path}/desc/r640_SP-nms4-7_scenes_{args.scene}-augNone_setlen1-dense.h5"
    out_path = f"{all_path}/mlp"
    os.makedirs(out_path, exist_ok=True)
    global_descriptor_dict  = load_dense_descriptors_to_mem(h5_file_path_list=[h5_path])

    dataset = DensePairInMemDataset(f"{all_path}/output_pairs/all_pairs_down.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, 
                           num_workers=args.num_workers, pin_memory=True)
    
    model = get_mlp(dim=args.dim).to(device)
    if args.load_ckpt is not None:
        ckpt = torch.load(args.load_ckpt)
        model.load_state_dict(ckpt)
        print(f'{args.load_ckpt} ckpt loaded')
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"{out_path}/runs/{timestamp}")
    ckpt_folder = f"{out_path}/ckpts/{timestamp}"
    os.makedirs(ckpt_folder, exist_ok=True)
    best_vloss = 1_000_000.

    for epoch in range(1, args.epochs + 1):
        print('EPOCH {}:'.format(epoch))
        start_time = time.time()
        model.train()
        # total_loss = 0
        for index, (f1, f2, label) in enumerate(trainloader):
            f1, f2, label = f1.to(device), f2.to(device), label.to(device)
            # breakpoint()

            z1, z2 = model(f1), model(f2)

            loss_contrast = contrastive_loss(z1, z2, label, margin=args.margin)
            loss_l2 = l2_loss(z1, z2, label)
            loss = loss_contrast + args.l2_weight * loss_l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # total_loss += loss.item()
            if index % 200 == 0:
                total_iteration = len(trainloader) * (epoch-1) + index
                writer.add_scalar("train/" + "total", loss.item(), total_iteration)
                writer.add_scalar("train/" + "L2", loss_l2.item(), total_iteration)
                writer.add_scalar("train/" + "contrast", loss_contrast.item(), total_iteration)
                
        print( "[E {} | iter {}] loss {}".format(epoch, index, loss.item()))
        
        val_loss, val_l2, val_contrast = do_evaluation_pair(args, model, valloader, device)
        
        print(f'[Validation]', val_loss.item())
        writer.add_scalar("val/" + 'total', val_loss.item(), total_iteration)
        writer.add_scalar("val/" + 'L2', val_l2.item(), total_iteration)
        writer.add_scalar("val/" + 'contrast', val_contrast.item(), total_iteration)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        end_time = time.time()
        epoch_time = end_time - start_time
        print('Time taken for one EPOCH {}: {:.2f} seconds'.format(epoch + 1, epoch_time))

        if val_loss < best_vloss:
            print(f'new ckpt get! val_loss:{val_loss}\n')
            best_vloss = val_loss
            checkpoint_path = f'{ckpt_folder}/epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            manage_checkpoints(ckpt_folder, max_checkpoints=10)
        scheduler.step(val_loss)



def main():
    parser = ArgumentParser()
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--l2_weight", type=float, default=0.1,
                    help="Weight for L2 loss on positive pairs")
    parser.add_argument("--scene", type=str)
    args = parser.parse_args()
    train(args)


def test():
    all_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train/sparse"
    h5_path = f"{all_path}/desc/r640_SP-k1024-nms4-7_scenes_scene_stairs-augNone_setlen1.h5"
    x = load_dense_descriptors_to_mem(h5_file_path_list=[h5_path])

    # npz_path = f"{all_path}/output_pairs/all_pairs_down.npz"
    # data = np.load(npz_path, allow_pickle=True)
    # img1 = data['img1']
    # img2 = data['img2']
    # xy1 = data['xy1']
    # xy2 = data['xy2']
    # label = data['label']
    breakpoint()



# python -m mlp.train_pair
# x['seq-06-frame-000497.color'].shape
# (1, 256, 60, 80)
if __name__=="__main__":
    test()
