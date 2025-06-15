import os
import torch
import h5py
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from .modules import get_mlp, MLP_module_16_short2
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
from mlp.tensor import batch_to_device
from mlp.mlp_utils import do_evaluation_pair2


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
                if 'descriptors' in h5[key]:
                    desc_np = h5[key]['descriptors'][:]
                    desc_np = desc_np.astype(np.float32)
                    desc_np = np.expand_dims(desc_np, axis=0)  # (1, 256, H', W')
                    feature_bank[key] = desc_np
    print(f"Loaded {len(feature_bank)} descriptors into memory.")
    return feature_bank


def pad_descriptors(descriptors, pad_value=0.0):
    max_len = max([d.shape[0] for d in descriptors])
    dim = descriptors[0].shape[1]
    padded = torch.full((len(descriptors), max_len, dim), pad_value, dtype=descriptors[0].dtype)
    mask = torch.zeros(len(descriptors), max_len, dtype=torch.bool)
    for i, d in enumerate(descriptors):
        padded[i, :d.shape[0]] = d
        mask[i, :d.shape[0]] = True
    return padded, mask


def collate_fn(batch):
    desc0_list, desc1_list, pos0_list, neg0_list, pos1_list, neg1_list = zip(*batch)

    # Pad desc0/desc1 for later use (可選)
    padded0, mask0 = pad_descriptors(desc0_list)
    padded1, mask1 = pad_descriptors(desc1_list)

    # 合併所有正負配對 (for training MLP)
    pos0 = torch.cat(pos0_list, dim=0)
    pos1 = torch.cat(pos1_list, dim=0)
    neg0 = torch.cat(neg0_list, dim=0)
    neg1 = torch.cat(neg1_list, dim=0)

    return {
        "desc0": padded0,
        "desc1": padded1,
        "mask0": mask0,
        "mask1": mask1,
        "pos0": pos0, "pos1": pos1,
        "neg0": neg0, "neg1": neg1
    }



class DensePairInMemDataset(Dataset):
    def __init__(self):
        all_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train/sparse"
        h5_path = f"{all_path}/desc/r640_SP-k1024-nms4-7_scenes_scene_stairs-augNone_setlen1.h5"
        
        my_dict = load_dense_descriptors_to_mem(h5_file_path_list=[h5_path])
        self.desc_all = []
        key_list = list(my_dict.keys())
        for key in key_list:
            self.desc_all.append(my_dict[key])
        all_matches = torch.load(f"{all_path}/all_values2.pt")
        # self.idx0_list = all_matches["idx0"]
        # self.idx1_list = all_matches["idx1"]
        self.desc0_list = all_matches["desc0"]
        self.desc1_list = all_matches["desc1"]
        self.m0_list = all_matches["m0"]
        self.match_tf_list = all_matches["match_true_false"]
        max_desc_len = 0
        leng = len(self.desc0_list)
        for i in range(leng):
            if (max_desc_len < len(self.desc0_list[i])):
                max_desc_len = len(self.desc0_list[i])
            if (max_desc_len < len(self.desc1_list[i])):
                max_desc_len = len(self.desc1_list[i])
        self.max_desc_len = max_desc_len
    def __len__(self): 
        return len(self.desc0_list)
    def __getitem__(self, idx):
        desc0 = self.desc0_list[idx]
        desc1 = self.desc1_list[idx]
        m0 = self.m0_list[idx]
        match_true_false = self.match_tf_list[idx]
        valid = (m0 > -1)
        m_desc0 = desc0[valid]
        # m_desc1 = desc1[m0[valid]]

        m_success_desc0 = m_desc0[match_true_false]
        m_fail_desc0 = m_desc0[~match_true_false]
        m_success_desc1 = desc1[m0[valid][match_true_false]]
        m_fail_desc1 = desc1[m0[valid][~match_true_false]]

        return desc0, desc1, m_success_desc0, m_fail_desc0, m_success_desc1, m_fail_desc1


def triplet_loss(f_pos0, f_pos1, f_neg0, f_neg1, margin=0.2):

    N = min(f_pos0.shape[0], f_neg1.shape[0], f_neg0.shape[0], f_pos1.shape[0])
    f_pos0 = f_pos0[:N]
    f_pos1 = f_pos1[:N]
    f_neg0 = f_neg0[:N]
    f_neg1 = f_neg1[:N]

    loss1 = F.relu(F.pairwise_distance(f_pos0, f_pos1) -
                   F.pairwise_distance(f_pos0, f_neg1) + margin)
    loss2 = F.relu(F.pairwise_distance(f_pos1, f_pos0) -
                   F.pairwise_distance(f_pos1, f_neg0) + margin)

    return (loss1.mean() + loss2.mean()) / 2


def manage_checkpoints(save_dir, max_checkpoints=10):
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoint_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    if len(checkpoint_files) > max_checkpoints:
        for old_ckpt in checkpoint_files[:-max_checkpoints]:
            old_ckpt_path = os.path.join(save_dir, old_ckpt)
            os.remove(old_ckpt_path)



def train(args):
    all_path = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/{args.scene}/train/sparse"
    out_path = f"{all_path}/mlp2"
    dataset = DensePairInMemDataset()
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers,
                             collate_fn=collate_fn, shuffle=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, 
                           collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    # data = dataset[0]
    # data = next(iter(dataloader))
    # model = get_mlp(dim=args.dim).to(device)
    model = MLP_module_16_short2().to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"{out_path}/runs/{timestamp}")
    ckpt_folder = f"{out_path}/ckpts/{timestamp}"
    os.makedirs(ckpt_folder, exist_ok=True)
    best_vloss = 1_000_000.
    mse_weight = 1.0

    for epoch in range(1, args.epochs + 1):
        print('EPOCH {}:'.format(epoch))
        start_time = time.time()
        model.train()
        for index, batch in enumerate(trainloader):
            # 壓縮器輸出
            batch = batch_to_device(batch, device)
            
            c_desc0, rec_desc0 = model(batch["desc0"])  # [B, L, D]
            c_desc1, rec_desc1 = model(batch["desc1"])

            # 只對有效位置計算重建 MSE
            mse0 = F.mse_loss(rec_desc0[batch["mask0"]], batch["desc0"][batch["mask0"]])
            mse1 = F.mse_loss(rec_desc1[batch["mask1"]], batch["desc1"][batch["mask1"]])
            mse_loss = (mse0 + mse1) / 2

            # 計算 triplet loss（使用 encoder 輸出的 compressed descriptor）
            c_pos0, _ = model(batch["pos0"])
            c_pos1, _ = model(batch["pos1"])
            c_neg0, _ = model(batch["neg0"])
            c_neg1, _ = model(batch["neg1"])
            loss_triplet = triplet_loss(c_pos0, c_pos1, c_neg0, c_neg1)
            loss = loss_triplet + mse_weight * mse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if index % 100 == 0:
                total_iteration = len(trainloader) * (epoch-1) + index
                writer.add_scalar("train/" + "total", loss.item(), total_iteration)
                writer.add_scalar("train/" + "L2", mse_loss.item(), total_iteration)
                writer.add_scalar("train/" + "loss_triplet", loss_triplet.item(), total_iteration)
            # breakpoint()
        print( "[E {} | iter {}] loss {}".format(epoch, index, loss.item()))
        val_loss, val_l2, val_contrast = do_evaluation_pair2(args, model, valloader, device)

        print(f'[Validation]', val_loss.item())
        writer.add_scalar("val/" + 'total', val_loss.item(), total_iteration)
        writer.add_scalar("val/" + 'L2', val_l2.item(), total_iteration)
        writer.add_scalar("val/" + 'loss_triplet', val_contrast.item(), total_iteration)
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


def test():
    all_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train/sparse"
    h5_path = f"{all_path}/desc/r640_SP-k1024-nms4-7_scenes_scene_stairs-augNone_setlen1.h5"
    x = load_dense_descriptors_to_mem(h5_file_path_list=[h5_path])
    y = torch.load(f"{all_path}/all_values.pt")


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
    parser.add_argument("--scene", type=str, 
                        default="scene_stairs")
    args = parser.parse_args()
    train(args)



# python -m mlp.train_pair2
if __name__=="__main__":
    main()
