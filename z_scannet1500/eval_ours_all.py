import os
import json
import pickle
import pprint
import argparse
from pathlib import Path
from itertools import chain
from datetime import datetime
from utils.match.comm import gather
from utils.match.metrics import aggregate_metrics
from render import feature_visualize_saving
import torch.utils.tensorboard as tensorboard
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def flattenList(x):
    return list(chain(*x))


def get_elapsed_time_for_tag(log_dir, tag_name):
    start_time = None
    end_time = None
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                file_path = os.path.join(root, file)
                event_acc = EventAccumulator(file_path)
                event_acc.Reload()
                if tag_name in event_acc.Tags().get('scalars', []):
                    events = event_acc.Scalars(tag_name)
                    start_time = datetime.fromtimestamp(events[0].wall_time) if start_time is None else start_time
                    end_time = datetime.fromtimestamp(events[-1].wall_time)
    if start_time and end_time:
        elapsed_time = end_time - start_time
        elapsed_time = int(elapsed_time.total_seconds())
        return elapsed_time
    else:
        return None


def compute_eval(all_path, save_path, out_name, match_name, eval_all=0, eval_pcd_size=0):
    all_path = all_path
    folders = os.listdir(all_path)
    folders = sorted(folders, key=lambda f: int(f[5:9]))
    folders = [os.path.join(all_path, f, f"sfm_sample/outputs/{out_name}/{match_name}/LG") for f in folders]
    aggregate_list = []
    gauss_list = {
        "ssim": 0,
        "lpips": 0,
        "psnr": 0,
    }
    total_elapsed_time = 0
    num = 0
    if eval_pcd_size:
        total_gauss_size = 0
    for idx, path in enumerate(folders):
        print(idx, path)
        with open(f"{path}/matching.pkl", 'rb') as f:
            p_file = pickle.load(f)
        aggregate_list.extend(p_file)
        if eval_all:
            run_paths = Path(path).parent.parent/"runs"
            run_folds = os.listdir(run_paths)
            run_folds = sorted(run_folds, key = lambda f:int(f.replace('_', '')))
            run_fold = run_folds[-1]
            run_path = run_paths/run_fold
            if os.path.exists(f"{path}/training_time.pkl"):
                with open(f"{path}/training_time.pkl", 'rb') as f:
                    elapsed_time = pickle.load(f)
            else:
                elapsed_time = get_elapsed_time_for_tag(run_path, "iter_time")
                with open(f"{path}/training_time.pkl", 'wb') as f:
                    pickle.dump(elapsed_time, f)
            total_elapsed_time = total_elapsed_time + elapsed_time
            with open(f"{path}/results.json", 'r') as f:
                r_file = json.load(f)
                gauss_list['ssim'] += r_file["ours_8000"]["SSIM"]
                gauss_list['psnr'] += r_file["ours_8000"]["PSNR"]
                gauss_list['lpips'] += r_file['ours_8000']['LPIPS']
            if eval_pcd_size:
                pcd_path = str(Path(path).parent.parent/"point_cloud"/"iteration_8000"/"point_cloud.ply")
                if os.path.exists(f"{path}/gs_size.pkl"):
                    with open(f"{path}/gs_size.pkl", 'rb') as f:
                        gs_size = pickle.load(f)
                else:
                    gs_size = os.path.getsize(pcd_path)/(1024 * 1024)
                    with open(f"{path}/gs_size.pkl", 'wb') as f:
                        pickle.dump(gs_size, f)
                total_gauss_size += gs_size
                num+=1
                print("size: ", gs_size)
            print("time: ", elapsed_time)

    metrics = {k: flattenList(gather(flattenList([_me[k] for _me in aggregate_list]))) for k in aggregate_list[0]}
    val_metrics_4tb = aggregate_metrics(metrics, 5e-4)
    if eval_all:
        avg_training_time = int(float(total_elapsed_time)/float(len(folders)))    
        gauss_list['ssim'] /= len(folders)
        gauss_list['psnr'] /= len(folders)
        gauss_list['lpips'] /= len(folders)
        if eval_pcd_size:
            avg_gs_size = total_gauss_size/ num
    
    
    pprint.pprint(val_metrics_4tb)
    if eval_all:
        pprint.pprint(gauss_list)
        print(avg_training_time)
        minutes, seconds = divmod(avg_training_time, 60)
        print(f"min:{minutes}, second:{seconds}")
        if eval_pcd_size:
            print(avg_gs_size)
    
    os.makedirs(save_path, exist_ok=True)
    file_name = out_name.replace(':', '_')
    save_path = f"{save_path}/{file_name}.txt"

    with open(f'{save_path}', 'a') as file:
        pp = pprint.PrettyPrinter(stream=file)
        pp.pprint(val_metrics_4tb)
        if eval_all:
            pp.pprint(gauss_list)
            if eval_pcd_size:
                print("gaussian size:", file=file)
                print(avg_gs_size, file=file)
            print("training time:", file=file)
            print(avg_training_time, file=file)
            print(f"min:{minutes}, second:{seconds}", file=file)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_path", type=str,)
    parser.add_argument("--save_path", type=str,)
    parser.add_argument("--out_name", type=str,)
    parser.add_argument("--match_name", type=str,)
    parser.add_argument("--eval_all", type=int,)
    parser.add_argument("--eval_pcd_size", type=int,)
    args = parser.parse_args()
    {    
        # all_path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test"
        # save_path = f"/home/koki/code/cc/feature_3dgs_2/z_result"
        # out_name = "SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_ScoreL2_ScoreScale0.6"
        # match_name = "MatchResult_KptKernalSize15_KptHist0.9_LGth0.01"
        # eval_all = 1
        # eval_pcd_size = 0
    }
    compute_eval(all_path=args.all_path, save_path=args.save_path, 
                 out_name=args.out_name, match_name=args.match_name, 
                 eval_all=args.eval_all, eval_pcd_size=args.eval_pcd_size)
