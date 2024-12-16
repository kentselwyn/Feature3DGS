import os
import argparse
import torch
from codes.used_codes.utils import load_image2
from encoders.superpoint.superpoint import SuperPoint
from encoders.disk_kornia import DISK
from encoders.aliked import ALIKED
from encoders.superpoint.mlp import get_mlp_model
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time



def plot_points(img: torch.Tensor, kpts: torch.Tensor):
    C, H, W = img.shape
    x_cods = kpts[:, 0]
    y_cods = kpts[:, 1]
    x_floor = torch.floor(x_cods).long()
    y_floor = torch.floor(y_cods).long()

    indices0 = [
        (y_floor, x_floor),
        (y_floor, x_floor + 1),
        (y_floor + 1, x_floor),
        (y_floor + 1, x_floor + 1)
    ]
    indices1 = [
        (y_floor - 1, x_floor - 1),
        (y_floor - 1, x_floor + 0),
        (y_floor - 1, x_floor + 1),
        (y_floor - 1, x_floor + 2),

        (y_floor + 0 ,x_floor - 1),
        (y_floor + 0, x_floor + 2),
        (y_floor + 1 ,x_floor - 1),
        (y_floor + 1, x_floor + 2),
        
        (y_floor + 2, x_floor - 1),
        (y_floor + 2, x_floor + 0),
        (y_floor + 2, x_floor + 1),
        (y_floor + 2, x_floor + 2)
    ]
    for (yi, xi) in indices0:
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        img[0, yi[valid], xi[valid]] = 0.9
        if C>1:
            img[1, yi[valid], xi[valid]] = 0.
            img[2, yi[valid], xi[valid]] = 0.
    for (yi, xi) in indices1:
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        img[0, yi[valid], xi[valid]] = 0.6
        if C>1:
            img[1, yi[valid], xi[valid]] = 0.
            img[2, yi[valid], xi[valid]] = 0.








def save_all(img: torch.Tensor, kpts: torch.Tensor, desc: torch.Tensor, sp_path: str, args):
            #  img_path: str, 
    img = img[0]
    kpts = kpts[0]

    _ , H, W = img.shape
    score = torch.zeros((1, H, W), dtype=torch.float32)

    if args.method=="ALIKED" or args.method=="DISK":
        desc = F.interpolate(desc.unsqueeze(0),  size=(int(desc.shape[1]/8), int(desc.shape[2]/8)), 
                             mode='bilinear', align_corners=True).squeeze(0)
    # plot_points(img, kpts)
    plot_points(score, kpts)

    img = img.permute(1,2,0).cpu().numpy()
    # plt.imsave(f'{img_path}.jpg', img)
    torch.save(desc, f"{sp_path}_fmap.pt")
    torch.save(score, f"{sp_path}_smap.pt")
    
    

    




def main(args):
    
    if args.method.startswith("SP"):
        conf = {
            "sparse_outputs": True,
            "dense_outputs": True,
            "max_num_keypoints": int(args.max_num_keypoints),
            "detection_threshold": args.th,
        }
        model = SuperPoint(conf).to("cuda").eval()
    elif args.method=="DISK":
        conf = {
            "dense_outputs": True,
            "max_num_keypoints": int(args.max_num_keypoints),
        }
        model = DISK(conf).to("cuda").eval()
    elif args.method=="ALIKED":
        conf = {
            "model_name": "aliked-n16",
            "max_num_keypoints": int(args.max_num_keypoints),
            "detection_threshold": 0.
        }
        model = ALIKED(conf).to("cuda").eval()
    
    mlp = get_mlp_model(dim = args.mlp_dim, type=args.method)
    
    mlp = mlp.to("cuda").eval()

    img_folder = f"{args.input}/{args.images}"
    # kptimg_folder = f"{args.input}/all_images/imrate:{args.resize_num}_th:{args.th}_mlpdim:{args.mlp_dim}"
    feature_folder = f"{args.input}/features/{args.feature_name}"
    

    target_images = [f for f in os.listdir(img_folder) if not os.path.isdir(os.path.join(img_folder, f))]
    target_images = [os.path.join(img_folder, f) for f in target_images]

    # os.makedirs(f"{args.input}/{args.image_folder}", exist_ok=True)
    os.makedirs(feature_folder, exist_ok=True)

    for t in target_images:
        print(f"Processing '{t}'...")
        img_name = t.split(os.sep)[-1].split(".")[0]

        resize_num = int(args.resize_num)

        img_tensor = load_image2(t, resize=resize_num).to("cuda").unsqueeze(0)
        data = {}
        data["image"] = img_tensor
        pred = model(data)

        desc = pred["dense_descriptors"][0]
        desc_mlp = mlp(desc.permute(1,2,0)).permute(2,0,1).contiguous().cpu()


        kpts = pred["keypoints"].cpu()

        # img_path = f"{args.input}/{args.image_folder}/{img_name}"
        sp_path = f"{feature_folder}/{img_name}"

        save_all(img_tensor, kpts, desc_mlp, sp_path, args)



        
# python dataset_build.py --input /home/koki/code/cc/feature_3dgs_2/img_match/Else/tandt_db/truck --resize_num 1 --max_num_keypoints 1024 --method SP_tank_db --images images_low_resolution
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_01/B",
    )
    parser.add_argument(
        "--resize_num",
        required=True,
    )
    parser.add_argument(
        "--method",
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
        "--images",
        type=str,
        default="images",
    )
    
    args = parser.parse_args()

    args.feature_name = f"{args.method}_imrate:{args.resize_num}_th:{args.th}_mlpdim:{args.mlp_dim}_kptnum:{int(args.max_num_keypoints)}_score0.6_{args.images}"
    
    main(args)

