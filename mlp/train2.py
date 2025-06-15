import os
from scene.colmap_loader import qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary2, \
    read_points3D_text2, read_points3D_nvm, read_extrinsics_text, read_intrinsics_text
from scene.colmap_loader import Image
from typing import Tuple, Dict, List

path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train"


def read_bin():
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    xyzs, rgbs, errors, tracks = read_points3D_binary2(bin_path)



def read_txt() -> Tuple[Dict[int, Image], List[List[Tuple[int, int]]]]:
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    xyzs, rgbs, errors, tracks = read_points3D_text2(txt_path)
    return cam_extrinsics, tracks


cam_extrinsics, tracks = read_txt()
valid_ids = set(cam_extrinsics.keys())


filtered_tracks = []
for track in tracks:
    filtered_track = [(img, idx) for img, idx in track if img in valid_ids]
    filtered_tracks.append(filtered_track)


def test():
    p3d_idx = 0
    p3d_kpt_idx = 0

    img_kpt = filtered_tracks[p3d_idx][p3d_kpt_idx]
    img_idx = img_kpt[0]
    kpt_idx = img_kpt[1]

    # breakpoint()
    kpt_coordinate = cam_extrinsics[img_idx].xys[kpt_idx]


positive_pairs = []
for track in filtered_tracks:
    if len(track) < 2:
        continue  # 至少兩張圖才可配對
    breakpoint()
    for i in range(len(track)):
        for j in range(i + 1, len(track)):
            img0, idx0 = track[i]
            img1, idx1 = track[j]
            positive_pairs.append((img0, idx0, img1, idx1))

im0 = positive_pairs[0][0]
id0 = positive_pairs[0][1]
im1 = positive_pairs[0][2]
id1 = positive_pairs[0][3]
# breakpoint()
print(cam_extrinsics[im0].xys[id0], cam_extrinsics[im1].xys[id1])

# for i in range(4):
    # print(positive_pairs[100][i], cam_extrinsics[0].xys[id0])
# breakpoint()


# python -m mlp.train2