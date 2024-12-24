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

path = Path(f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_chess/train/rgb")
data_name = path.parts[-3]

random.seed(0)

    
class MLP_sp_data_scannet(Dataset):
    def __init__(self, conf) -> None:
        # glob = ["*.jpg", "*.png"]
        # g = glob[0]
        conf = OmegaConf.create(conf)
        images = list(conf.path.glob("**/" + "*.png"))
        # images = [path for path in images_ if 'all_images' not in path.parts]
        random.shuffle(images)
        
        # s_sample = random.sample(s_images, 50000)

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
resize = 1024
configs = {
    "sp": {
        "name": f"r{resize}_SP-k{n_kpts}-nms4-{data_name}",
        "keys": ["keypoints", "descriptors", ],
        "gray": True,
        "conf": {
            "name": "superpoint",
            "nms_radius": 4,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
            "force_num_keypoints": True,
        },
    },
    "sift": {
        "name": f"r{resize}_SIFT-k{n_kpts}",
        "keys": ["keypoints", "descriptors" ],
        "gray": True,
        "conf": {
            "name": "sift",
            "max_num_keypoints": n_kpts,
            "options": {
                "peak_threshold": 0.001,
            },
            "peak_threshold": 0.001,
            "device": "cpu",
            "force_num_keypoints": True,
        },
    },
}



def run_export(args, feature_file):
    conf = {
        "data":{
            "path": path,
            "preprocessing": {
                'resize': resize,
                'side': 'long',
                'square_pad': False,
            },
        },
        "model": configs[args.method],
    }
    conf = OmegaConf.create(conf)
    keys = configs[args.method]["keys"]
    dataset = MLP_sp_data_scannet(conf.data)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate, num_workers=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SuperPoint(conf.model.conf).eval().to(device)
    # breakpoint()

    export_predictions(loader, model, feature_file, as_half=True, keys=keys)



data_path = f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/{data_name}/desc_data"


# python -m mlp.export --method sp
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="sp")
    args = parser.parse_args()
    export_name = configs[args.method]["name"]
    feature_file = Path(
            data_path, export_name + ".h5"
    )
    run_export(args, feature_file=feature_file)





