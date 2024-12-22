import torch
from pathlib import Path
import importlib
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_gt_data():
    gt_data = {"gt_matches0": torch.rand([4, 512]),
            "gt_matches1": torch.rand([4, 512]),
            "gt_assignment": torch.rand([4, 512, 512]),}
    return gt_data

def get_spp_data():
    data = {"image": torch.rand([4,640,480]),
            "image_size": torch.rand([4,2])}
            
    return data




def count_trainable_params(model):
    count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return count


def count_all_parameters(model):
    count = sum(param.numel() for param in model.parameters())
    return count





def copy_module():
    module = "core.models.my_matchers"
    # mod_dir = Path(__import__(str(module)).__file__)
    # imported_module = importlib.import_module(module)
    mod_dir = Path(importlib.import_module(module).__file__).parent
    print(mod_dir)
    shutil.copytree(mod_dir, Path("/home/koki/gluetrain/demo") / module, dirs_exist_ok=True)

