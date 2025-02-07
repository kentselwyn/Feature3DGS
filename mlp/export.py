import torch
import random
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from .base_dataset import collate
from torch.utils.data import Dataset, DataLoader
from .image import ImagePreprocessor, load_image
from .export_predictions import export_predictions
from encoders.superpoint.superpoint import SuperPoint


random.seed(0)
class MLP_sp_data_scannet(Dataset):
    def __init__(self, conf) -> None:
        conf = OmegaConf.create(conf)
        images = list(conf.path.glob("**/" + "*.png"))
        random.shuffle(images)
        self.images = images
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = load_image(path=path)
        data = self.preprocessor(img)
        data['name'] = str(path)
        return data
    
    def __len__(self):
        return len(self.images)
    

n_kpts = 1024
resize = 640
extractor_configs = {
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


def run_export(args, data_path, feature_file):
    conf = {
        "data":{
            "path": data_path,
            "preprocessing": {
                'resize': resize,
                'side': 'long',
                'square_pad': False,
            },
        },
        "model": extractor_configs[args.method],
    }
    conf = OmegaConf.create(conf)
    keys = extractor_configs[args.method]["keys"]
    dataset = MLP_sp_data_scannet(conf.data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate, num_workers=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperPoint(conf.model.conf).eval().to(device)
    export_predictions(loader, model, feature_file, as_half=True, keys=keys)


# python -m mlp.export --method sp
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="SP")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--scene_name", type=str)
    parser.add_argument("--dense", action='store_true')
    args = parser.parse_args()
    name = f"{args.data_name}_{args.scene_name}"

    if args.dense:
        export_name = f"r{resize}_{args.method}-k{n_kpts}-nms4-{name}-dense"
        keys = ["keypoints", "descriptors", "dense_descriptors"]
    else:
        export_name = f"r{resize}_{args.method}-k{n_kpts}-nms4-{name}"
        keys = ["dense_descriptors"]
    
    data_name = args.data_name
    scene_name = args.scene_name
    data_path = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/{data_name}/{scene_name}/train/rgb"
    out_path = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/{data_name}/{scene_name}/desc_data"

    feature_file = Path(out_path, export_name + ".h5")
    run_export(args, data_path=data_path, feature_file=feature_file)
