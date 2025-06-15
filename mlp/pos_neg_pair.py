import os
from scene.colmap_loader import qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary2, \
    read_points3D_text2, read_points3D_nvm, read_extrinsics_text, read_intrinsics_text
from scene.colmap_loader import Image
from typing import Tuple, Dict, List
from collections import defaultdict
import re
from argparse import ArgumentParser
import random
import numpy as np


def read_txt(path) -> Tuple[Dict[int, Image], List[List[Tuple[int, int]]]]:
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    images = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    xyzs, rgbs, errors, tracks, pid_to_xyz = read_points3D_text2(txt_path)
    return images, tracks, xyzs, pid_to_xyz



def extract_frame_id(name):
    match = re.search(r'frame-(\d+)', name)
    return int(match.group(1)) if match else -1



def extract_seq(image_name):
    # 假設格式為：seq-XX/frame-YYYYYY.color.png
    match = re.match(r'(seq-\d+)/frame-(\d+)\.color\.png', image_name)
    if match:
        sequence = match.group(1)         # e.g., 'seq-03'
        frame_id = int(match.group(2))    # e.g., 276
        return sequence
    return None



def generate_positives(pid_to_obs, images_dict, max_pairs_per_point=10):
    xy_positive_pairs = []

    for pid, obs in pid_to_obs.items():
        pairs = []
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                (img1, idx1, img1_name) = obs[i]
                (img2, idx2, img2_name) = obs[j]
                if extract_seq(img1_name) != extract_seq(img2_name):
                    xy1 = images_dict[img1].xys[idx1]
                    xy2 = images_dict[img2].xys[idx2]
                    pairs.append(((img1, xy1, img1_name), (img2, xy2, img2_name)))
                    continue
                if abs(extract_frame_id(img1_name) - extract_frame_id(img2_name)) <= 100:
                    continue
                xy1 = images_dict[img1].xys[idx1]
                xy2 = images_dict[img2].xys[idx2]
                pairs.append(((img1, xy1, img1_name), (img2, xy2, img2_name)))
        
        # 抽樣最多 max_pairs_per_point 筆
        if len(pairs) > max_pairs_per_point:
            pairs = random.sample(pairs, max_pairs_per_point)
        
        xy_positive_pairs.extend(pairs)
    # breakpoint()

    return xy_positive_pairs



def generate_positives2(pid_to_obs, images_dict, ):
    key, value = next(iter(images_dict.items()))
    xy_positive_pairs = []

    for pid, obs in pid_to_obs.items():
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                (img1, idx1, img1_name) = obs[i]
                (img2, idx2, img2_name) = obs[j]
                xy1 = images_dict[img1].xys[idx1]
                xy2 = images_dict[img2].xys[idx2]
                if abs(extract_frame_id(img1_name)-extract_frame_id(img2_name)) >100\
                    or extract_seq(img1_name)!=extract_seq(img2_name):
                    xy_positive_pairs.append(((img1, xy1, img1_name), (img2, xy2, img2_name)))
    return xy_positive_pairs



def generate_negatives(
    xyzs,                      # shape: [N, 3], 每個 point3D_id 對應的 3D 座標
    pid_to_xyz, 
    pid_to_obs,              
    images_dict,               # Dict[image_id] → Image
    mode='soft',               # 'soft' or 'hard'
    num_negatives=100000,      # 負樣本數量
    dist_threshold=1.0         # soft 時候兩個 3D 點的最小距離
):
    neg_pairs = []
    pid_list = list(pid_to_obs.keys())
    tried = 0

    if mode == 'soft':
        while len(neg_pairs) < num_negatives and tried < num_negatives * 10:
            tried += 1
            pid1, pid2 = random.sample(pid_list, 2)
            p1, p2 = pid_to_xyz[pid1], pid_to_xyz[pid2]
            if np.linalg.norm(p1 - p2) < dist_threshold:
                continue

            obs1 = random.choice(pid_to_obs[pid1])
            obs2 = random.choice(pid_to_obs[pid2])
            img1, idx1, name1 = obs1
            img2, idx2, name2 = obs2
            xy1 = images_dict[img1].xys[idx1]
            xy2 = images_dict[img2].xys[idx2]
            neg_pairs.append(((img1, xy1, name1), (img2, xy2, name2)))

    elif mode == 'hard':
        image_ids = list(images_dict.keys())
        for image_id in image_ids:
            image = images_dict[image_id]
            pid_idxs = [(idx, pid) for idx, pid in enumerate(image.point3D_ids) if pid != -1]
            if len(pid_idxs) < 2:
                continue
            for i in range(len(pid_idxs)):
                for j in range(i + 1, len(pid_idxs)):
                    idx1, pid1 = pid_idxs[i]
                    idx2, pid2 = pid_idxs[j]
                    if pid1 == pid2:
                        continue
                    xy1 = image.xys[idx1]
                    xy2 = image.xys[idx2]
                    name = image.name
                    neg_pairs.append(((image_id, xy1, name), (image_id, xy2, name)))
                    if len(neg_pairs) >= num_negatives:
                        break
                if len(neg_pairs) >= num_negatives:
                    break
            if len(neg_pairs) >= num_negatives:
                break
    else:
        raise ValueError("mode must be 'soft' or 'hard'")
    print(f"Generated {len(neg_pairs)} {mode} negative pairs.")
    return neg_pairs



def save_pairs_to_npz(pairs, filename):
    img1_list, xy1_list, img2_list, xy2_list = [], [], [], []
    for (img1, xy1, name1), (img2, xy2, name2) in pairs:
        img1_list.append(name1)
        img2_list.append(name2)
        xy1_list.append(xy1)
        xy2_list.append(xy2)
    np.savez(filename,
             img1=img1_list,
             xy1=np.array(xy1_list),
             img2=img2_list,
             xy2=np.array(xy2_list))



def save_all_npz(out_name, all_path):
    def load_npz(path):
        data = np.load(path, allow_pickle=True)
        return data['img1'], data['xy1'], data['img2'], data['xy2']
    # 讀入三個檔案
    img1_p, xy1_p, img2_p, xy2_p = load_npz(f'{all_path}/pos.npz')
    img1_s, xy1_s, img2_s, xy2_s = load_npz(f'{all_path}/neg_soft.npz')
    img1_h, xy1_h, img2_h, xy2_h = load_npz(f'{all_path}/neg_hard.npz')
    # 合併
    img1 = np.concatenate([img1_p, img1_s, img1_h])
    xy1  = np.concatenate([xy1_p,  xy1_s,  xy1_h ])
    img2 = np.concatenate([img2_p, img2_s, img2_h])
    xy2  = np.concatenate([xy2_p,  xy2_s,  xy2_h ])
    # 建立 label（1=pos, 0=neg）
    label = np.concatenate([
        np.ones(len(img1_p), dtype=np.int64),
        np.zeros(len(img1_s), dtype=np.int64),
        np.zeros(len(img1_h), dtype=np.int64)
    ])
    # 儲存為新 npz
    np.savez(f"{all_path}/{out_name}.npz", img1=img1, xy1=xy1, img2=img2, xy2=xy2, label=label)
    print("合併完成，總數：", len(img1))



# python -m mlp.pos_neg_pair --scene
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--scene", type=str)
    args = parser.parse_args()
    path = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/{args.scene}/train"
    images_dict, _, xyzs, pid_to_xyz = read_txt(path)
    pid_to_obs = defaultdict(list)

    for image_id, image in images_dict.items():
        for idx, pid in enumerate(image.point3D_ids):
            if pid != -1:
                pid_to_obs[pid].append((image_id, idx, image.name))

    pos = generate_positives(pid_to_obs, images_dict, )
    print(len(pos))
    # breakpoint()
    neg_soft = generate_negatives(xyzs, pid_to_xyz, pid_to_obs, images_dict, 
                                  mode='soft', num_negatives=len(pos)*4/7, dist_threshold=1.0)
    neg_hard = generate_negatives(xyzs, pid_to_xyz, pid_to_obs, images_dict, 
                                  mode='hard', num_negatives=len(pos)*3/7)
    save_folder = f"{path}/sparse/output_pairs"
    os.makedirs(save_folder, exist_ok=True)
    save_pairs_to_npz(pos, f"{save_folder}/pos.npz")
    save_pairs_to_npz(neg_soft, f"{save_folder}/neg_soft.npz")
    save_pairs_to_npz(neg_hard, f"{save_folder}/neg_hard.npz")
    save_all_npz("all_pairs_down", all_path=save_folder)
