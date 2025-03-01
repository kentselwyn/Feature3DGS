import os
import shutil
from pathlib import Path


def remove_images():
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
def change_folders_name():
    path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test_1500_renders"
    folders = os.listdir(path)
    for name in folders:
        color_path = f"{path}/{name}/color"
        os.rename(color_path, f"{path}/{name}/images")


def remove_unused_folders():
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


# python -m z_scannet1500.tmp
# ls -l /home/koki/code/cc/feature_3dgs_2/scannet/test | grep '^d' | wc -l
# find /path/to/directory -maxdepth 1 -type f | wc -l
if __name__=="__main__":
    change_folders_name()
