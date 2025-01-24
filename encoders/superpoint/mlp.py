import torch
from torch import nn
from pathlib import Path
from torchrl.modules import MLP



class MLP_module_4_short(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=4   , num_cells=[128, 64, 32, 16, 8])
        self.MLP_de = MLP(in_features=4,   out_features=256,  num_cells=[8, 16, 32, 64, 128])
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back

class MLP_module_8_short(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=8   , num_cells=[128, 64, 32, 16, 8])
        self.MLP_de = MLP(in_features=8,   out_features=256,  num_cells=[8, 16, 32, 64, 128])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        # desc_back = self.MLP_de(desc_mlp)
        return desc_mlp
    
    def decode(self, desc_mlp: torch.Tensor):
        desc_back = self.MLP_de(desc_mlp)
        return desc_back

class MLP_module_16_short(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=16   , num_cells=[128, 64, 32, 16])
        self.MLP_de = MLP(in_features=16, out_features=256,    num_cells=[16, 32, 64, 128])
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        # desc_back = self.MLP_de(desc_mlp)
        return desc_mlp
    def decode(self, desc_mlp: torch.Tensor):
        desc_back = self.MLP_de(desc_mlp)
        return desc_back






class MLP_module_8_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=128, out_features=8   , num_cells=[64, 32, 16, 8])
        self.MLP_de = MLP(in_features=8,   out_features=128,  num_cells=[8, 16, 32, 64])
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        # desc_back = self.MLP_de(desc_mlp)
        return desc_mlp
    def decode(self, desc_mlp: torch.Tensor):
        desc_back = self.MLP_de(desc_mlp)
        return desc_back
    def back(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back


class MLP_module_16_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=128, out_features=16   , num_cells=[64, 32, 16, 16])
        self.MLP_de = MLP(in_features=16, out_features=128,    num_cells=[16, 16, 32, 64])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        # desc_back = self.MLP_de(desc_mlp)
        return desc_mlp
    def decode(self, desc_mlp: torch.Tensor):
        desc_back = self.MLP_de(desc_mlp)
        return desc_back
    def back(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back


# "/mnt/home_6T/public/koki/ckpt/mlp/Superpoint/"

CKPT_FOLDER = Path("/mnt/home_6T/public/koki/mlpckpt/ckpt/")

def get_mlp_model(dim = 16, type = "SP"):
    if type=="SP_scannet":
        name = "SP_scannet"
        if dim==8:
            model_path = CKPT_FOLDER/name/"epoch_142.pt"
            model = MLP_module_8_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==16:
            model_path = CKPT_FOLDER/name/"dim16_epoch_288.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if type=="SP_homo_pair":
        name = "SP_homo_pair"
        if dim==8:
            model_path = CKPT_FOLDER/"SP_homo_pair"/"epoch_241.pt"
            model = MLP_module_8_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==16:
            model_path = CKPT_FOLDER/name/"dim16_epoch_1099.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if type=="SP_tank_db":
        name = "SP_tank_db"
        if dim==16:
            model_path = CKPT_FOLDER/name/"epoch_999_train_from_scrach.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if type=="SP_drjohnson_skpt":
        if dim==16:
            model_path = CKPT_FOLDER/type/"epoch_992.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if type=="SP_drjohnson":
        name = "SP_drjohnson"
        if dim==16:
            model_path = CKPT_FOLDER/name/"epoch_997.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if type=="SP_playroom":
        name = "SP_playroom"
        if dim==16:
            model_path = CKPT_FOLDER/name/"epoch_998.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if type=="SP_train":
        name = "SP_train"
        if dim==16:
            model_path = CKPT_FOLDER/name/"epoch_991.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if type=="SP_truck":
        name = "SP_truck"
        if dim==16:
            model_path = CKPT_FOLDER/name/"epoch_993.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if type=="SP":
        name = "SP"
        if dim==8:
            model_path = CKPT_FOLDER/name/"short_pair_8/model_20240221_105112_498"
            model = MLP_module_8_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==16:
            model_path = CKPT_FOLDER/name/"short_pair_16/model_20240221_105233_496"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    elif type=="DISK":
        name = "DISK"
        if dim==8:
            model_path = CKPT_FOLDER/name/"time:20241022_004619_dim8_batch64_lr5e-05_epoch10000_2/epoch_6260.pt"
            model = MLP_module_8_128()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==16:
            model_path = CKPT_FOLDER/name/"time:20241021_133114_dim16_batch64_lr0.0001_epoch10000_2/epoch_9959.pt"
            model = MLP_module_16_128()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    elif type=="ALIKED":
        name = "ALIKED"
        if dim==8:
            model_path = CKPT_FOLDER/name/"time:20241022_011835_dim8_batch64_lr5e-05_epoch10000_2/epoch_3825.pt"
            model = MLP_module_8_128()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==16:
            model_path = CKPT_FOLDER/name/"time:20241021_134201_dim16_batch64_lr5e-05_epoch10000_2/epoch_9991.pt"
            model = MLP_module_16_128()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    
    return model
        


def get_mlp_dataset(dim=16, dataset="pgt_7scenes_chess"):
    CKPT_FOLDER = Path(f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc")
    if dataset=="pgt_7scenes_chess":
        if dim==4:
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/type:SP_time:20241224_223934_dim4_batch64_lr0.0008_epoch5000/epoch_542.pt"
            model = MLP_module_4_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        elif dim==8:
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/type:SP_time:20241224_213513_dim8_batch64_lr0.001_epoch10000/epoch_9813.pt"
            model = MLP_module_8_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        elif dim==16:
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/type:SP_time:20241224_210047_dim16_batch64_lr0.001_epoch10000/epoch_9810.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="pgt_7scenes_fire":
        if dim==16:
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/type:SP_time:20250107_162819_dim16_batch64_lr0.0008_epoch5000/epoch_1776.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    
    if dataset=="pgt_7scenes_heads":
        if dim==16:
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/type:SP_time:20250115_024821_dim16_batch64_lr0.0008_epoch5000/epoch_2335.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="pgt_7scenes_office":
        if dim==16:
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/type:SP_time:20250115_030334_dim16_batch64_lr0.0008_epoch5000/epoch_791.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="pgt_7scenes_pumpkin":
        if dim==16:
            "type:SP_time:20250115_031231_dim16_batch64_lr0.0008_epoch5000"
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/epoch_852.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="pgt_7scenes_redkitchen":
        if dim==16:
            "type:SP_time:20250115_032624_dim16_batch64_lr0.0008_epoch5000"
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/epoch_456.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    
    if dataset=="pgt_7scenes_stairs":
        if dim==16:
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/type:SP_time:20250107_160254_dim16_batch64_lr0.0008_epoch5000/epoch_1722.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    
    if dataset=="Cambridge_KingsCollege":
        if dim==16:
            model_path = CKPT_FOLDER/f"Cambridge/{dataset}/mlpckpt/type:SP_time:20250107_161544_dim16_batch64_lr0.0008_epoch5000/epoch_2753.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="Cambridge_OldHospital":
        if dim==16:
            model_path = CKPT_FOLDER/f"Cambridge/{dataset}/mlpckpt/type:SP_time:20250107_162033_dim16_batch64_lr0.0008_epoch5000/epoch_4096.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    
    if dataset=="Cambridge_ShopFacade":
        if dim==16:
            model_path = CKPT_FOLDER/f"Cambridge/{dataset}/mlpckpt/type:SP_time:20250115_034958_dim16_batch64_lr0.0008_epoch5000/epoch_4982.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="Cambridge_StMarysChurch":
        if dim==16:
            model_path = CKPT_FOLDER/f"Cambridge/{dataset}/mlpckpt/type:SP_time:20250115_035230_dim16_batch64_lr0.0008_epoch5000/epoch_2317.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    return model



# python -m encoders.superpoint.mlp
if __name__=="__main__":
    model = get_mlp_dataset(16)

    breakpoint()


