import os
import cv2
import argparse
import numpy as np
import torch
from utils.utils import load_image2
from encoders.superpoint.superpoint import SuperPoint
from encoders.superpoint.mlp import get_mlp_model
import matplotlib.pyplot as plt
import time



def save_kptimg(img: torch.Tensor, kpts: torch.Tensor, 
                kptimg_path: str, bimg_path: str):
    
    _, _, H, W = img.shape
    img = img[0].cpu().numpy().transpose(1,2,0)
    kpts = kpts[0].cpu().numpy()

    start = time.time()

    bimg = np.zeros([H, W, 3])
    plt.imshow(bimg)
    plt.scatter(kpts[:,0], kpts[:,1], c="red", s=1, marker="o")
    plt.axis('off')
    plt.savefig(f"{bimg_path}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    plt.imshow(img)
    plt.scatter(kpts[:,0],kpts[:,1], c="red", s=1, marker="o")
    plt.axis('off')
    plt.savefig(f"{kptimg_path}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    end = time.time()

    print(f"process time: {end-start}")



def Save_all(img: torch.Tensor, kpts: torch.Tensor, desc: torch.Tensor, 
             kptimg_path: str, bimg_path: str, sp_path: str):
    save_kptimg(img, kpts, kptimg_path, bimg_path)
    torch.save(desc, f"{sp_path}_fmap_CxHxW.pt")




def main(args):
    conf = {
        "sparse_outputs": True,
        "max_num_keypoints": 512,
        "detection_threshold": 0.05,
    }
    model = SuperPoint(conf).to("cuda").eval()
    mlp = get_mlp_model()
    mlp = mlp.to("cuda").eval()

    img_folder = f"{args.input}/all_images/images"
    kptimg_folder = f"{args.input}/all_images/image_kpts"
    bimg_folder = f"{args.input}/all_images/b_image_kpts"
    sp_folder = f"{args.input}/features/sp_feature"
    

    target_images = [f for f in os.listdir(img_folder) if not os.path.isdir(os.path.join(img_folder, f))]
    target_images = [os.path.join(img_folder, f) for f in target_images]

    os.makedirs(kptimg_folder, exist_ok=True)
    os.makedirs(bimg_folder, exist_ok=True)
    os.makedirs(sp_folder, exist_ok=True)

    for t in target_images:
        print(f"Processing '{t}'...")
        img_name = t.split(os.sep)[-1].split(".")[0]
        img_tensor = load_image2(t).to("cuda").unsqueeze(0)
        data = {}
        data["image"] = img_tensor
        pred = model(data)
        desc = mlp(pred["descriptors"]).cpu()
        kpts = pred["keypoints"].cpu()

        kptimg_path = f"{kptimg_folder}/{img_name}"
        bimg_path = f"{bimg_folder}/{img_name}"
        sp_path = f"{sp_folder}/{img_name}"

        Save_all(img_tensor, kpts, desc, kptimg_path, bimg_path, sp_path)

    
    for t in target_images:
        print(f"Processing second '{t}'...")
        img_name = t.split(os.sep)[-1].split(".")[0]
        bimg_path = f"{bimg_folder}/{img_name}.png"
        sp_path = f"{sp_folder}/{img_name}"
        R_channel_tensor = load_image2(bimg_path)[0].unsqueeze(0)
        torch.save(R_channel_tensor, f"{sp_path}_smap_CxHxW.pt")



        
# python -m codes.used_codes.dataset_build
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_00/B",
    )
    args = parser.parse_args()
    
    main(args)
