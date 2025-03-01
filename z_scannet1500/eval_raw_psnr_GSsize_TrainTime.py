import os
import json
import pickle
import pprint
from pathlib import Path
from z_scannet1500.eval_ours_all import get_elapsed_time_for_tag


def compute_eval(out_name = "raw_imrate:1"):
    all_path = "/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test"
    folders = os.listdir(all_path)
    folders = sorted(folders, key= lambda f:int(f[5:9]))
    folders = [os.path.join(all_path, f, f"sfm_sample/outputs/{out_name}/match_result_superpoint_image_renders/LG") for f in folders]
    gauss_list = {
        "ssim": 0,
        "lpips": 0,
        "psnr": 0,
    }
    gauss_size = 0
    total_elapsed_time = 0

    for path in folders:
        print(path)
        with open(f"{path}/results.json", 'r') as f:
            r_file = json.load(f)
            gauss_list['ssim'] += r_file["ours_8000"]["SSIM"]
            gauss_list['psnr'] += r_file["ours_8000"]["PSNR"]
            gauss_list['lpips'] += r_file['ours_8000']['LPIPS']
        gs_size = os.path.getsize(str(Path(path).parent.parent/"point_cloud"/"iteration_8000"/"point_cloud.ply"))/(1024 * 1024)

        gauss_size += gs_size
        run_paths = Path(path).parent.parent/"runs"
        run_folds = os.listdir(run_paths)
        run_folds = sorted(run_folds, key = lambda f:int(f.replace('_', '')))

        run_fold = run_folds[-1]
        run_path = run_paths/run_fold

        if os.path.exists(f"{path}/training_time.pkl"):
            with open(f"{path}/training_time.pkl", 'rb') as f:
                elapsed_time = pickle.load(f)
        else:
            elapsed_time = get_elapsed_time_for_tag(run_path, "train_loss_patches/total_loss")
            with open(f"{path}/training_time.pkl", 'wb') as f:
                pickle.dump(elapsed_time, f)
        total_elapsed_time = total_elapsed_time + elapsed_time
    
    avg_training_time = int(float(total_elapsed_time)/float(len(folders)))
    gauss_list['ssim'] /= len(folders)
    gauss_list['psnr'] /= len(folders)
    gauss_list['lpips'] /= len(folders)
    avg_gs_size = gauss_size/ len(folders)

    print(avg_gs_size)
    print(avg_training_time)
    minutes, seconds = divmod(avg_training_time, 60)
    print(minutes, seconds)
    pprint.pprint(gauss_list)

    save_folder = f"/home/koki/code/cc/feature_3dgs_2/z_result"
    os.makedirs(save_folder, exist_ok=True)
    file_name = out_name.replace(':', '_')
    save_path = f"{save_folder}/{file_name}.txt"

    with open(f'{save_path}', 'a') as file:
        pp = pprint.PrettyPrinter(stream=file)
        pp.pprint(gauss_list)

        print("gaussian size:", file=file)
        print(avg_gs_size, file=file)
        
        print("training time:", file=file)
        print(avg_training_time, file=file)
        print(f"min:{minutes}, second:{seconds}", file=file)


# python compute_eval_image.py
if __name__=="__main__":
    compute_eval()
