import os
import time
import shutil
import pickle
import torch
import argparse
import subprocess
from tqdm import tqdm
import z_scannet1500.dataset_build as dataset_build
import eval.metrics_gauss as metrics_gauss
from eval.eval_scannet1500 import match_eval






def move_colmap_bin():
    s_path = "/home/koki/code/cc/feature_3dgs_2/sample"
    s_folds = os.listdir(s_path)
    s_folds = sorted(s_folds, key=lambda f: int(f[5:9]))
    t_path = "/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test"
    t_folds = os.listdir(t_path)
    t_folds = sorted(t_folds, key=lambda f: int(f[5:9]))

    s_folds = [os.path.join(s_path, f) for f in s_folds]
    t_folds = [os.path.join(t_path, f, 'sfm_sample') for f in t_folds]

    for idx, fold in enumerate(tqdm(s_folds)):
        t_folder = f"{t_folds[idx]}/sparse/0"

        if os.path.exists(t_folder):
            if len(os.listdir(t_folder))>=3:
                continue
        os.makedirs(t_folder, exist_ok=True)
        # files = os.listdir(fold)
        for f in os.listdir(fold):
            source_file = os.path.join(fold, f)
            if os.path.isfile(source_file):
                # Move the file to the destination directory
                if not os.path.exists(os.path.join(t_folder, f)):
                    shutil.move(source_file, t_folder)





def run_scannet(start, end, args, test_list=None):
    all_path = "/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test"
    folders = os.listdir(all_path)
    folders = sorted(folders, key=lambda f: int(f[5:9]))
    folders = [os.path.join(all_path, f, 'sfm_sample') for f in folders]

    if test_list is None:
        if end is not None:
            folders = folders[start: end]
        else:
            folders = folders[start:]
    else:
        tmp = []
        for id in test_list:
            tmp.append(folders[id])
        folders = tmp

    feature_name = f"{args.method}_imrate:{args.resize_num}_th:{args.th}_mlpdim:{args.mlp_dim}_kptnum:{int(args.max_num_keypoints)}_score:{args.score_scale}"

    if args.resize_num == 1:
        args.image_folder = "images"
    else:
        args.image_folder = f"images_s{args.resize_num}"


    args.feature_name = feature_name

    aggregate_list = []
    gauss_result = []
    traing_time = []


    save_path = f"/home/koki/code/cc/feature_3dgs_2/img_match/result/{args.method}_dim{args.mlp_dim}"
    os.makedirs(save_path, exist_ok=True)

    for fold_path in folders:
        print(f"processing {fold_path}...")
        scene_num = int(fold_path.split('/')[-2][5:9])
        args.input = fold_path

        torch.cuda.empty_cache()
        if not os.path.exists(f"{args.input}/features/{args.feature_name}"):
            dataset_build.main(args)
        elif len(os.listdir(f"{args.input}/features/{args.feature_name}"))==0:
            dataset_build.main(args)

        torch.cuda.empty_cache()
        gaussian_path = f"{args.input}/outputs/{args.feature_name}/point_cloud/iteration_8000/point_cloud.ply"
        if not os.path.exists(gaussian_path):
            st = time.time()
            command = ['bash', 'zenith_scripts/train.sh', args.input, args.feature_name, args.image_folder, args.score_scale]
            subprocess.run(command, check=True)
            en = time.time()
            traing_time.append((scene_num, en-st))
            os.makedirs(f'{args.input}/outputs/{args.feature_name}/{args.match_name}/LG', exist_ok=True)
            with open(f'{args.input}/outputs/{args.feature_name}/{args.match_name}/LG/Training_time.pkl', 'wb') as file:
                pickle.dump((scene_num, en-st), file)


        torch.cuda.empty_cache()


        if not os.path.exists(f"{args.input}/outputs/{args.feature_name}/rendering/pairs/ours_8000/score_tensors"):
            command = ['bash', 'zenith_scripts/render.sh', args.input, args.feature_name, args.image_folder]
            subprocess.run(command, check=True)
        elif len(os.listdir(f"{args.input}/outputs/{args.feature_name}/rendering/pairs/ours_8000/score_tensors"))==0:
            command = ['bash', 'zenith_scripts/render.sh', args.input, args.feature_name, args.image_folder]
            subprocess.run(command, check=True)

        

        torch.cuda.empty_cache()
        tmp_list = match_eval(args)
        aggregate_list.extend(tmp_list)

        tmp_g_result = metrics_gauss.evaluate([f"{args.input}/outputs/{args.feature_name}/rendering"], args)
        gauss_result.append((scene_num, tmp_g_result))
        
        
        if os.path.exists(f"{args.input}/features/{args.feature_name}"):
            shutil.rmtree(f"{args.input}/features/{args.feature_name}")

        torch.cuda.empty_cache()
    
    
    now = time.time()

    with open(f'{save_path}/matching_{start}_{end}_{now}.pkl', 'wb') as file:
        pickle.dump(aggregate_list, file)
    
    with open(f'{save_path}/Gauss_perform_{start}_{end}_{now}.pkl', 'wb') as file:
        pickle.dump(gauss_result, file)

    with open(f'{save_path}/Training_time_{start}_{end}_{now}.pkl', 'wb') as file:
        pickle.dump(traing_time, file)
    






# 0721
# 0754
# 0753
# 0776
# 0715
# 0711 out mem
# 30, 30, 20, 20
# python run_img_match.py --mlp_dim 16 --method DISK
# python run_img_match.py --mlp_dim 16 --method ALIKED
# python run_img_match.py --mlp_dim 16 --method SP_homo
# python run_img_match.py --mlp_dim 16 --method SP_scannet
# python run_img_match.py --mlp_dim 8 --method SP_scannet
# python run_img_match.py --mlp_dim 16 --method SP --histogram_th 0.95 --resize_num 2
# python run_img_match.py --mlp_dim 16 --method SP --resize_num 2
# python run_img_match.py --mlp_dim 8 --method SP_scannet --resize_num 2
# python run_img_match.py --mlp_dim 16 --method SP --score_scale 0
if __name__=="__main__":

    start = 707-707
    end = 752-707

    # r_list = [721, 754, 776]
    # r_list = [r-707 for r in r_list]


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resize_num",
        default=1,
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--th",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max_num_keypoints",
        type=float,
        default=1024,
    )
    parser.add_argument(
        "--match_name",
        default="match_result",
    )
    

    parser.add_argument(
        "--method",
        required=True,
    )
    parser.add_argument(
        "--kernel_size",
        default=7
    )
    parser.add_argument(
        "--score_kpt_th",
        default=0.02
    )
    parser.add_argument(
        "--histogram_th",
        default=None
    )
    parser.add_argument(
        "--score_scale",
        default=None
    )
    args = parser.parse_args()

    run_scannet(start, end ,args, test_list=None)




