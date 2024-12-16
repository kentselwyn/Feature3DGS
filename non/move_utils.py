import os
import shutil
import argparse
from codes.used_codes.utils import load_image2
from encoders.superpoint.superpoint import SuperPoint
from encoders.superpoint.mlp import get_mlp_model
from dataset_build import save_all



def move_folders(source_path, folders):
    for name in folders:
        name_path = os.path.join(source_path, name)
        for x in ["A", "B"]:
            img_path = os.path.join(name_path, f"{x}/all_images")
            out_path = os.path.join(name_path, f"{x}/outputs")
            
            os.makedirs(img_path, exist_ok=True)
            os.makedirs(os.path.join(name_path, f"{x}/features"), exist_ok=True)
            os.makedirs(out_path,  exist_ok=True)

            ip = os.path.join(name_path, f"{x}/images")
            if os.path.exists(ip):
                os.rename(ip, os.path.join(img_path, "img1296x968_raw"),)
            
            op = os.path.join(name_path, f"{x}/output")
            if os.path.exists(op):
                os.rename(op, os.path.join(out_path, "output"))




def build_img_feature(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type=str,)
    parser.add_argument("--images", "-i", type=str)
    parser.add_argument("--kpt_name", "-k", type=str)

    parser.add_argument("--img_resize", "-ir", type=int, default=1)
    parser.add_argument("--mlp_dim", "-md", type=int, default=16)
    # for superpoint
    parser.add_argument("--num_kpt", "-n", type=int, default=1024)
    parser.add_argument("--th", type=float, default=0.01)

    args = parser.parse_args()

    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": args.num_kpt,
        "detection_threshold": args.th,
    }
    encoder = SuperPoint(conf).to("cuda").eval()
    mlp = get_mlp_model(dim = args.mlp_dim)
    mlp = mlp.to("cuda").eval()

    img_folder = f"{args.source}/all_images/{args.images}"
    kptimg_folder = f"{args.source}/all_images/{args.kpt_name}"
    sp_folder = f"{args.source}/features/{args.kpt_name}"

    target_images = [f for f in os.listdir(img_folder) if not os.path.isdir(os.path.join(img_folder, f))]
    target_images = [os.path.join(img_folder, f) for f in target_images]

    os.makedirs(kptimg_folder, exist_ok=True)
    os.makedirs(sp_folder, exist_ok=True)

    for t in target_images:
        print(f"Processing '{t}'...")
        img_name = t.split(os.sep)[-1].split(".")[0]
        img_tensor = load_image2(t, resize=args.img_resize).to("cuda").unsqueeze(0)

        data = {}
        data["image"] = img_tensor
        pred = encoder(data)

        desc = pred["dense_descriptors"][0]
        desc_mlp = mlp(desc.permute(1,2,0)).permute(2,0,1).contiguous().cpu()


        kpts = pred["keypoints"].cpu()

        img_path = f"{kptimg_folder}/{img_name}"
        sp_path = f"{sp_folder}/{img_name}"

        save_all(img_tensor, kpts, desc_mlp, img_path, sp_path)




def resize_img(source_path, folders, resize=[648, 484]):
    from PIL import Image
    for name in folders:
        name_path = os.path.join(source_path, name)
        for x in ["A", "B"]:
            img_folder = os.path.join(name_path, f"{x}/all_images/img1296x968_raw")
            target_images = [f for f in os.listdir(img_folder)]
            target_images = [os.path.join(img_folder, f) for f in target_images]

            resize_path = os.path.join(os.path.dirname(img_folder), "img648x484_raw")
            os.makedirs(resize_path, exist_ok=True)

            for t in target_images:
                print(f"Processing '{t}'...")
                img_resize = Image.open(t).resize(resize)

                img_name = t.split(os.sep)[-1].split(".")[0]
                img_resize.save(f"{resize_path}/{img_name}.jpg")



def build_img_feature_allfolders(source_path, folders):
    for name in folders:
        for x in ['A', 'B']:
            name_path = os.path.join(source_path, f"{name}/{x}")
            image_path = os.path.join(name_path, "all_images")
            feature_path = os.path.join(name_path, "features")

            img_folders = os.listdir(image_path)
            feature_folders = os.listdir(feature_path)
            image_len = len(os.listdir(os.path.join(image_path, "img648x484_raw")))
            for i_fold in img_folders:
                f_fold = i_fold[:len(i_fold)-3]+"feature"
                if i_fold[-3:] != "raw":
                    continue
                
                # check if there is any elements in f_fold
                if f_fold in feature_folders:
                    if len(os.listdir(os.path.join(feature_path, f_fold))) == (image_len*2):
                        continue
                
                raw_i_path = os.path.join(image_path, i_fold)
                i_path = os.path.join(image_path, f_fold)
                f_path = os.path.join(feature_path, f_fold)
                os.makedirs(i_path, exist_ok=True)
                os.makedirs(f_path, exist_ok=True)

                extract_kpts(raw_i_path, i_path, f_path)

                

def extract_kpts(img_folder, out_img, out_feature, resize_rate=None):
    conf = {
        "sparse_outputs": True,
        "dense_outputs": True,
        "max_num_keypoints": 1024,
        "detection_threshold": 0.01,
    }
    encoder = SuperPoint(conf).to("cuda").eval()
    mlp = get_mlp_model(dim = 16)
    mlp = mlp.to("cuda").eval()

    target_images = [f for f in os.listdir(img_folder) if not os.path.isdir(os.path.join(img_folder, f))]
    target_images = [os.path.join(img_folder, f) for f in target_images]
    for t in target_images:
        print(f"Processing '{t}'...")
        img_name = t.split(os.sep)[-1].split(".")[0]
        img_tensor = load_image2(t, resize=resize_rate).to("cuda").unsqueeze(0)

        data = {}
        data["image"] = img_tensor
        pred = encoder(data)

        desc = pred["dense_descriptors"][0]
        desc_mlp = mlp(desc.permute(1,2,0)).permute(2,0,1).contiguous().cpu()
        kpts = pred["keypoints"].cpu()

        img_path = f"{out_img}/{img_name}"
        sp_path = f"{out_feature}/{img_name}"

        save_all(img_tensor, kpts, desc_mlp, img_path, sp_path)

        

                
def rename_folder():
    source_path = "/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test"
    folders = os.listdir(source_path)
    folders = [os.path.join(source_path, f, "test_pairs") for f in folders]

    for path in folders:
        if os.path.exists(f"{path}/color"):
            os.rename(f"{path}/color", f"{path}/images")
                


def compute_eval():
    all_path = "/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test"
    folders = os.listdir(all_path)
    folders = [os.path.join(all_path, f, "sfm_sample/outputs/SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6/match_result/LG") for f in folders]
    
    print(folders[0])



def remove_sfm0():
    all_path = "/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test"
    folders = os.listdir(all_path)
    folders = [os.path.join(all_path, f, "sfm0") for f in folders]
    # print(folders[0])
    for path in folders:
        print(path)
        if os.path.exists(path):
            shutil.rmtree(path)



# python move_utils.py
if __name__=="__main__":
    remove_sfm0()

