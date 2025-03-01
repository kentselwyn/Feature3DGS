import os
import json
import pickle
import pprint
from pathlib import Path
from itertools import chain
from datetime import datetime
from utils.comm import gather
from utils.metrics import aggregate_metrics
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


def compute_eval(all_path, out_name = "SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_score0.6"):
    all_path = all_path
    folders = os.listdir(all_path)
    folders = [os.path.join(all_path, f, f"sfm_sample/outputs/{out_name}/match_result/LG") for f in folders]
    aggregate_list = []
    gauss_list = {
        "ssim": 0,
        "lpips": 0,
        "psnr": 0,
    }
    total_gauss_size = 0
    total_elapsed_time = 0
    num = 0
    error_list = []
    for path in folders:
        print(path)
        with open(f"{path}/matching.pkl", 'rb') as f:
            p_file = pickle.load(f)
        aggregate_list.extend(p_file)
        with open(f"{path}/results.json", 'r') as f:
            r_file = json.load(f)
            gauss_list['ssim'] += r_file["ours_8000"]["SSIM"]
            gauss_list['psnr'] += r_file["ours_8000"]["PSNR"]
            gauss_list['lpips'] += r_file['ours_8000']['LPIPS']
        gs_size = os.path.getsize(str(Path(path).parent.parent/"point_cloud"/"iteration_8000"/"point_cloud.ply"))/(1024 * 1024)
        total_gauss_size += gs_size
        num+=1
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
        print("size: ", gs_size)
        print("time: ", elapsed_time)

    avg_training_time = int(float(total_elapsed_time)/float(len(folders)))    
    gauss_list['ssim'] /= len(folders)
    gauss_list['psnr'] /= len(folders)
    gauss_list['lpips'] /= len(folders)
    avg_gs_size = total_gauss_size/ num
    metrics = {k: flattenList(gather(flattenList([_me[k] for _me in aggregate_list]))) for k in aggregate_list[0]}
    val_metrics_4tb = aggregate_metrics(metrics, 5e-4)
    
    pprint.pprint(val_metrics_4tb)
    pprint.pprint(gauss_list)
    print(avg_gs_size)
    print(avg_training_time)
    minutes, seconds = divmod(avg_training_time, 60)
    print(f"min:{minutes}, second:{seconds}")

    save_folder = f"/home/koki/code/cc/feature_3dgs_2/z_result"
    os.makedirs(save_folder, exist_ok=True)
    file_name = out_name.replace(':', '_')
    save_path = f"{save_folder}/{file_name}.txt"

    with open(f'{save_path}', 'a') as file:
        pp = pprint.PrettyPrinter(stream=file)
        pp.pprint(val_metrics_4tb)
        pp.pprint(gauss_list)
        print("gaussian size:", file=file)
        print(avg_gs_size, file=file)
        print("training time:", file=file)
        print(avg_training_time, file=file)
        print(f"min:{minutes}, second:{seconds}", file=file)


# SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_score0.6
# SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6
# SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_score0.6
# SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_score0.6
# python -m z_scannet1500.eval_ours_all
if __name__=="__main__":
    all_path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test"
    out_name = "SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_score0.6"
    compute_eval(all_path=all_path, out_name=out_name)
