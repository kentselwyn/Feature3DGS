import os
import shutil
from pathlib import Path


def RemoveImages():
    path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test_1500_renders"
    folders = os.listdir(path)
    for name in folders:
        color_path = f"{path}/{name}/color"
        images = os.listdir(color_path)
        for img in images:
            img_p = f"{color_path}/{img}"
            if Path(img_p).suffix == ".jpg":
                if os.path.exists(img_p):
                    os.remove(img_p)


# SP_scannet_imrate:2_th:0.01_mlpdim:8_kptnum:1024_score0.6
# SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_score0.6
def RenameFolders():
    # pp = "SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024"
    old_name = "images"
    new_name = "color"
    all_path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test_1500_renders"
    folders = os.listdir(all_path)
    for fold in folders:
        old_path = f"{all_path}/{fold}/{old_name}"
        new_path = f"{all_path}/{fold}/{new_name}"
        os.rename(old_path, new_path)


def RemoveUnusedFolders():
    path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test_1500_renders"
    folders = os.listdir(path)
    for name in folders:
        depth_path = f"{path}/{name}/depth"
        intrinsic_path = f"{path}/{name}/intrinsic"
        pose_path = f"{path}/{name}/pose"
        if os.path.exists(depth_path):
            shutil.rmtree(depth_path)
            shutil.rmtree(intrinsic_path)
            shutil.rmtree(pose_path)


# SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_score0.6
# SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6
# SP_scannet_imrate:2_th:0.01_mlpdim:8_kptnum:1024_score0.6
# SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_score0.6
def RemoveUnusedPcd(out_name="SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_score0.6"):
    path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test"
    folders = os.listdir(path)
    for name in folders:
        pcd_path = f"{path}/{name}/sfm_sample/outputs/{out_name}/point_cloud/iteration_8000/point_cloud.ply"
        if os.path.exists(pcd_path):
            print(pcd_path)
            os.remove(pcd_path)


def RemoveUnusedOutfolder(out_name="SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_Scoreweighted_ScoreScale0.6"):
    path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test"
    folders = os.listdir(path)
    for name in folders:
        fold_path = f"{path}/{name}/sfm_sample/outputs/{out_name}"
        if os.path.exists(fold_path):
            print(fold_path)
            shutil.rmtree(fold_path)


def FindAllFolders_ChangeNames():
    root_dir = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test"
    old_name = "MatchResult_KptKernalSize$15_ScoreL2_KptHist0.9_LGth0.01"
    new_name = "MatchResult_KptKernalSize15_KptHist0.9_LGth0.01"
    for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname == old_name:
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, new_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                else:
                    print(f"Skipping: {old_path} (target name exists)")


def FindAllFiles_ChangeNames():
    root_dir = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test"
    old_name = "Training_time.pkl"
    new_name = "training_time.pkl"
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == old_name:
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                else:
                    print(f"Skipping: {old_path} (target name exists)")


def CopyFiles():
    root_dir = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test"
    target_name = "MatchResult_KptKernalSize15_KptHist0.9_LGth0.01"
    for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname == target_name:
                target_path = os.path.join(dirpath, dirname, "LG")
                print(target_path)
                target_path = Path(target_path)
                parent_path = target_path.parent.parent
                copy_path = parent_path/"match_result/LG"
                name = "training_time.pkl"
                file = copy_path/name
                # s_result = copy_path/"results.json"
                if not os.path.exists(target_path/name) and os.path.exists(file):
                    shutil.copy(file, target_path/name)
                    # shutil.copy(s_result, target_path/"results.json")


def test():
    import pickle
    all_path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test/scene0799_00/sfm_sample/outputs"
    with open(f"{all_path}/SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_Scoreweighted_ScoreScale0.6/MatchResult_KptKernalSize15_KptHist0.9_LGth0.01/LG/training_time.pkl", 'rb') as f:
        elapsed_time = pickle.load(f)
        print(elapsed_time)


def RenameFiles():
    # old exps
    # "match_result/LG"
    # "Training_time" -> "SceneNum_TrainingTime"
    # new exps
    # "MatchResult_KptKernalSize15_KptHist0.9_LGth0.01/LG"
    # "training_time" -> "SceneNum_TrainingTime"
    root_dir = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test"
    folders = os.listdir(root_dir)
    for fold in folders:
        path = f"{root_dir}/{fold}/sfm_sample/outputs/"
        for exp in new_exps:
            exp_path = f"{path}/{exp}/MatchResult_KptKernalSize15_KptHist0.9_LGth0.01/LG/training_time.pkl"
            new_name_path = f"{path}/{exp}/MatchResult_KptKernalSize15_KptHist0.9_LGth0.01/LG/SceneNum_TrainingTime.pkl"
            if os.path.exists(exp_path):
                os.rename(exp_path, new_name_path)



def CopyFolders():
    src_path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test_1500"
    tar_path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test_1500_renders"
    folders = os.listdir(src_path)
    for fold in folders:
        src_pp = f"{src_path}/{fold}"
        sd_p = f"{src_pp}/depth"
        si_p = f"{src_pp}/intrinsic"
        sp_p = f"{src_pp}/pose"
        tar_pp = f"{tar_path}/{fold}"
        td_p = f"{tar_pp}/depth"
        ti_p = f"{tar_pp}/intrinsic"
        tp_p = f"{tar_pp}/pose"
        shutil.copytree(sd_p, td_p)
        shutil.copytree(si_p, ti_p)
        shutil.copytree(sp_p, tp_p)



def main():
    RenameFiles()


old_exps = ["SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_ScoreL2_ScoreScale0.6", "SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6", 
            "SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6", "SP_scannet_imrate:2_th:0.01_mlpdim:8_kptnum:1024_score0.6"]

new_exps = ["SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_Scoreweighted_ScoreScale0.6", "SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_Scoreweighted_ScoreScale0.6", 
            "SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_ScoreL2_ScoreScale0.6", "SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_Scoreweighted_ScoreScale0.6",
            "SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_Scoreweighted_ScoreScale0.6", ]

# python -m z_scannet1500.tmp
# ls -l /home/koki/code/cc/feature_3dgs_2/scannet/test | grep '^d' | wc -l
# find /path/to/directory -maxdepth 1 -type f | wc -l
if __name__=="__main__":
    RemoveUnusedPcd("SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_ScoreL2_ScoreScale0.6")
    RemoveUnusedPcd("SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_Scoreweighted_ScoreScale0.6")
    RemoveUnusedPcd("SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6")
    RemoveUnusedPcd("SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_Scoreweighted_ScoreScale0.6")

    RemoveUnusedPcd("SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_ScoreL2_ScoreScale0.6")
    RemoveUnusedPcd("SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_Scoreweighted_ScoreScale0.6")
    RemoveUnusedPcd("SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6")
    RemoveUnusedPcd("SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_Scoreweighted_ScoreScale0.6")
