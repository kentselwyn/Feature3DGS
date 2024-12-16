import os
import numpy as np
import shutil
import time
import subprocess
from plyfile import PlyData
from tqdm import tqdm
from omegaconf import OmegaConf
from scene.gaussian_model import GaussianModel
from dataset_build import save_all
from encoders.superpoint.superpoint import SuperPoint
from encoders.superpoint.mlp import get_mlp_model
from codes.used_codes.utils import load_image2





def visualize_gaussian():
    input_path = '/home/koki/code/cc/feature_3dgs_2/all_data/scene0713_00/A/outputs/0/input.ply'
    gaussian_path = '/home/koki/code/cc/feature_3dgs_2/all_data/scene0713_00/A/outputs/0/point_cloud/iteration_10000/point_cloud.ply'


    ply_input = PlyData.read(input_path)
    ply_gaussian = PlyData.read(gaussian_path)
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(gaussian_path)

    print(ply_gaussian.elements[0])
    print(ply_input.elements[0]['red'].shape)
    print(ply_gaussian.elements[0]['x'].shape)

    sh_feature = gaussians.get_features
    seman_feature = gaussians.get_semantic_feature
    score_feature = gaussians.get_score_feature

    print(sh_feature.shape)
    print(sh_feature[0])
    print(seman_feature.shape)
    print(seman_feature[0])
    print(score_feature.shape)
    print(score_feature[0:10])




def read_test_trans():
    path = "/home/koki/code/cc/feature_3dgs_2/scannet/test_transformations.npz"
    trans = dict(np.load(path, allow_pickle=True))
    transformations_dict = trans['transformations'].item()
    print(transformations_dict.keys())
    print(transformations_dict['ref_transformations_list'].keys())

    print(transformations_dict['ref_transformations_list']['scene0762_00'])
    print(transformations_dict['src_transformations_list']['scene0762_00'])
    print(transformations_dict['gt_transformations_list']['scene0762_00'])
    



def remove_unused_folder(scannet_train_path, scannet_test_path, train_filtered, test_filtered):
    for folder in tqdm(train_filtered):
        for t in ['A', 'B']:
            path = f'{scannet_train_path}/{folder}/{t}'

            out_folder = f"{path}/output"
            f_parh = f'{path}/features.h5'
            m_path = f'{path}/matches.h5'
            p_path = f'{path}/pairs-netvlad.txt'
            g_path = f'{path}/global-feats-netvlad.h5'

            if os.path.exists(out_folder):
                shutil.rmtree(out_folder)

            remove_list = [f_parh, m_path, p_path, g_path]
            [os.remove(x) for x in remove_list if os.path.exists(x)]


    for folder in tqdm(test_filtered):
        for t in ['A', 'B']:
            path = f'{scannet_test_path}/{folder}/{t}'

            out_folder = f"{path}/output"
            f_parh = f'{path}/features.h5'
            m_path = f'{path}/matches.h5'
            p_path = f'{path}/pairs-netvlad.txt'
            g_path = f'{path}/global-feats-netvlad.h5'

            if os.path.exists(out_folder):
                shutil.rmtree(out_folder)

            remove_list = [f_parh, m_path, p_path, g_path]
            [os.remove(x) for x in remove_list if os.path.exists(x)]




def run_gaussian_train(SOURSE_PATH, img_folder, feature_folder, out_name, score_loss, conf):
    OUT_PATH = f"{SOURSE_PATH}/outputs/{out_name}"

    if not conf.train.use_1080:
        python_command = [
            "python", "train.py",
            "-s", SOURSE_PATH,
            "-m", OUT_PATH,
            "-i", img_folder,
            "-f", feature_folder,
            "--iterations", "10000",
            "--score_loss", score_loss
        ]
    else:
        python_command = [
            "python", "train_b4.py",
            "-s", SOURSE_PATH,
            "-m", OUT_PATH,
            "-i", img_folder,
            "-f", feature_folder,
            "--iterations", "10000",
            "--score_loss", score_loss
        ]
    subprocess.run(python_command, check=True)
    current_script_path = __file__
    shutil.copy(current_script_path, OUT_PATH)



def main(conf, folder_paths):
    model = SuperPoint(conf.extractor).to("cuda").eval()
    mlp = get_mlp_model(dim = conf.mlp.dim).to("cuda").eval()

    for index, folder in enumerate(folder_paths):
        for x in ['A', 'B']:
            path = os.path.join(folder, x)
            print(f"Processing '{path}'...")
            print(f'Now process {index}_{x}/{len(folder_paths)}')
            
            img_folder = os.path.join(path, 'images')
            kp_folder = os.path.join(path, 'kpt_images')
            sp_folder = os.path.join(path, 'features')

            target_images = [f for f in os.listdir(img_folder) if not os.path.isdir(os.path.join(img_folder, f))]
            target_images = [os.path.join(img_folder, f) for f in target_images]

            out_folder = os.path.join(path, 'outputs')
            os.makedirs(out_folder, exist_ok=True)
            os.makedirs(kp_folder, exist_ok=True)
            os.makedirs(sp_folder, exist_ok=True)

            ply_path = os.path.join(out_folder, "0/point_cloud/iteration_10000/point_cloud.ply")
            if os.path.exists(ply_path):
                print('ply exists, pass')
                continue

            start = time.time()
            for t in target_images:
                img_name = t.split(os.sep)[-1].split(".")[0]
                img_tensor = load_image2(t, resize=conf.data.resize_rate).to("cuda").unsqueeze(0)
                data = {}
                data["image"] = img_tensor
                pred = model(data)
                desc = pred["dense_descriptors"][0]
                
                desc_mlp = mlp(desc.permute(1,2,0)).permute(2,0,1).contiguous().cpu()
                kpts = pred["keypoints"].cpu()

                kp_path = f"{kp_folder}/{img_name}"
                sp_path = f"{sp_folder}/{img_name}"

                save_all(img_tensor, kpts, desc_mlp, kp_path, sp_path)
            end = time.time()
            print(f"time: {end-start}")

            run_gaussian_train(SOURSE_PATH=path, img_folder="kpt_images", feature_folder="features", out_name="0", 
                               score_loss=conf.gaussian.score_loss, conf=conf)

            shutil.rmtree(kp_folder)
            shutil.rmtree(sp_folder)




def split_list(lst, parts):
    # Calculate the size of the first (parts - 1) chunks
    chunk_size = len(lst) // parts
    out = []
    for i in range(parts - 1):
        out.append(lst[i*chunk_size: (i + 1)*chunk_size])
    out.append(lst[(parts - 1) * chunk_size:])

    return out





# python dataset_build_all.py
if __name__=="__main__":
    scannet_train_path = "/home/koki/code/cc/feature_3dgs_2/scannet/train"
    scannet_test_path = "/home/koki/code/cc/feature_3dgs_2/scannet/test"

    removed_data = ['scene0000_00', 'scene0000_01', 'scene0708_00', 'scene0713_00', 'scene0724_00', 'scene0755_00', ]
                     

    train_scenes = os.listdir(scannet_train_path)
    test_scenes = os.listdir(scannet_test_path)

    train_filtered = [item for item in train_scenes if item not in removed_data]
    test_filtered = [item for item in test_scenes if item not in removed_data]

    train_filtered = sorted(train_filtered, key=lambda x: (int(x[5:9]), int(x[10:12])))
    test_filtered = sorted(test_filtered, key=lambda x: (int(x[5:9]), int(x[10:12])))


    train_paths = [os.path.join(scannet_train_path, x) for x in train_filtered]
    test_paths = [os.path.join(scannet_test_path, x) for x in test_filtered]


    conf = OmegaConf.load('/home/koki/code/cc/feature_3dgs_2/conf/0_dataset_build_b4.yaml')
    # all_paths = train_paths + test_paths

    # print(len(train_paths))
    # print(len(test_paths))
    # print(len(all_paths))

    # parts = 6
    # out = split_list(all_paths, parts)

    # remove0 = ['scene0002_01', 'scene0002_00', 'scene0000_02', 'scene0001_01']
    # remove1 = ['scene0124_00', 'scene0124_01', 'scene0125_00', 'scene0126_00']
    # remove0 = [os.path.join(scannet_train_path, x) for x in remove0]
    # remove1 = [os.path.join(scannet_train_path, x) for x in remove1]

    # r0 = out[0] # for b6
    # r0 = [item for item in r0 if item not in remove0]
    # r1 = out[1] # for b5
    # r1 = [item for item in r1 if item not in remove1]
    # r2 = out[2] # for b4 0
    # r3 = out[3] # for b3 1
    # r4 = out[4] # for b4 1
    # r5 = out[5]
    parts = 2
    out = split_list(test_paths, parts)
    r0 = out[0]
    r1 = out[1]

    main(conf, r1)





