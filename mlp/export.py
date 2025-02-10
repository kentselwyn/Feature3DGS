import torch
import random
import argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from .base_dataset import collate
from matplotlib import pyplot as plt
from mlp.augmentations import augmentations
from torch.utils.data import Dataset, DataLoader
from .image import ImagePreprocessor, read_image
from .export_predictions import export_predictions
from encoders.superpoint.superpoint import SuperPoint

random.seed(0)
torch.manual_seed(0)
n_kpts = 1024
resize = 640
class MLP_sp_data_scannet(Dataset):
    def __init__(self, conf) -> None:
        conf = OmegaConf.create(conf)
        self.conf = conf
        images = list(conf.path.glob("**/" + "*.png"))
        self.images = images * conf.multiple
        self.images = sorted(self.images, key=lambda p: str(p))
        self.preprocessor = ImagePreprocessor(conf.preprocessing)
        if conf.augment=="sequence":
            seq_lst = set([p.parts[-1][:6] for p in self.images])
            seq_lst = sorted(seq_lst, key=lambda p: str(p))
            self.augments = [augmentations[f"{idx}"] for idx in range(len(seq_lst))]
            self.seq_dict = dict(zip(seq_lst, self.augments))
        else:
            self.augment = augmentations[conf.augment]() if conf.augment else augmentations["identity"]()
    def __getitem__(self, idx):
        path = self.images[idx]
        img = read_image(path)
        if self.conf.augment != "sequence":
            img_a = self.augment(img, return_tensor=True)
        else:
            seq_name = path.parts[-1][:6]
            img_a = self.seq_dict[seq_name](img, return_tensor=True)
        data = self.preprocessor(img_a)
        data['name'] = str(path) + str(idx)
        if self.conf.save_aug_img:
            plt.imsave(f"/home/koki/code/cc/feature_3dgs_2/test_imgs/{path.name}.png", img_a.permute(1,2,0).numpy())
        return data
    def __len__(self):
        return len(self.images)


model_configs = {
    "SP": {
        "gray": True,
        "conf": {
            "name": "superpoint",
            "nms_radius": 4,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
            "force_num_keypoints": True,
        },
    },
}


def run_export(args, keys, data_path, feature_file):
    conf = {
        "data":{
            "path": data_path,
            "preprocessing": {
                'resize': resize,
                'side': 'long',
                'square_pad': False,
            },
            "multiple": args.multiple,
            "augment": args.augment,
            "save_aug_img":args.img_save,
        },
        "model": model_configs[args.method],
    }
    conf = OmegaConf.create(conf)
    dataset = MLP_sp_data_scannet(conf.data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperPoint(conf.model.conf).eval().to(device)
    for index, data_ in enumerate(loader):
        print(index)
    # export_predictions(loader, model, feature_file, as_half=True, keys=keys)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="SP")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--scene_name", type=str)
    parser.add_argument("--augment", type=str)
    parser.add_argument("--multiple", type=int, default=4)
    parser.add_argument("--dense", action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_save", action="store_true")
    args = parser.parse_args()
    name = f"{args.data_name}_{args.scene_name}"
    if args.dense:
        export_name = f"r{resize}_{args.method}-k{n_kpts}-nms4-{name}-dense"
        keys = ["dense_descriptors"]
    else:
        export_name = f"r{resize}_{args.method}-k{n_kpts}-nms4-{name}-aug{args.augment}_setlen{args.multiple}"
        keys = ["descriptors", ]
    data_name = args.data_name
    scene_name = args.scene_name
    data_path = Path(f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/{data_name}/{scene_name}/train/rgb")
    out_path = Path(f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/{data_name}/{scene_name}/desc_data")
    feature_file = Path(out_path, export_name + ".h5")
    run_export(args, keys=keys, data_path=data_path, feature_file=feature_file)
