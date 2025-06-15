import torch
from torch import nn
from pathlib import Path
from torchrl.modules import MLP



class MLP_module_4_short(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=4   , num_cells=[128, 64, 32, 16, 8])
        self.MLP_de = MLP(in_features=4,   out_features=256,  num_cells=[8, 16, 32, 64, 128])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        # desc_back = self.MLP_de(desc_mlp)
        return desc_mlp
    
    def decode(self, desc_mlp: torch.Tensor):
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

class MLP_module_32_short(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=32   , num_cells=[128, 64, 32])
        self.MLP_de = MLP(in_features=32,   out_features=256,  num_cells=[32, 64, 128])
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        # desc_back = self.MLP_de(desc_mlp)
        return desc_mlp
    def decode(self, desc_mlp: torch.Tensor):
        desc_back = self.MLP_de(desc_mlp)
        return desc_back


class MLP_module_64_short(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=64   , num_cells=[128, 64, 64])
        self.MLP_de = MLP(in_features=64,   out_features=256,  num_cells=[64, 64, 128])
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
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/epoch_852.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="pgt_7scenes_redkitchen":
        if dim==16:
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
        if dim==32:
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/type:SP_time:20250203_064553_dim32_batch64_lr0.0008_epoch5000/epoch_1212.pt"
            model = MLP_module_32_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==64:
            model_path = CKPT_FOLDER/f"7_scenes/{dataset}/mlpckpt/type:SP_time:20250203_065135_dim64_batch64_lr0.0008_epoch5000/epoch_1722.pt"
            model = MLP_module_64_short()
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
    
    if dataset=="Cambridge":
        if dim==16:
            model_path = CKPT_FOLDER/f"Cambridge/mlpckpt/type:SP_time:20250204_235348_dim16_batch64_lr0.0008_epoch5000/epoch_1015.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==32:
            model_path = CKPT_FOLDER/f"Cambridge/mlpckpt/type:SP_time:20250204_234918_dim32_batch64_lr0.0008_epoch5000/epoch_910.pt"
            model = MLP_module_32_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==64:
            model_path = CKPT_FOLDER/f"Cambridge/mlpckpt/type:SP_time:20250204_235020_dim64_batch64_lr0.0008_epoch5000/epoch_981.pt"
            model = MLP_module_64_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    
    if dataset=="all":
        if dim==4:
            model_path = CKPT_FOLDER/f"mlpckpt/type:SP_time:20250211_153429_dim4_batch64_lr0.0008_epoch5000/epoch_150.pt"
            model = MLP_module_4_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==8:
            model_path = CKPT_FOLDER/f"mlpckpt/type:SP_time:20250211_154225_dim8_batch64_lr0.0008_epoch5000/epoch_143.pt"
            model = MLP_module_8_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==16:
            model_path = CKPT_FOLDER/f"mlpckpt/type:SP_time:20250211_154406_dim16_batch64_lr0.0008_epoch5000/epoch_156.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="all_augmy":
        if dim==4:
            model_path = CKPT_FOLDER/f"mlpckpt/type:SP_time:20250211_221717_dim4_batch64_lr0.0008_epoch5000_-augmyaug_setlen2/epoch_71.pt"
            model = MLP_module_4_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==8:
            model_path = CKPT_FOLDER/f"mlpckpt/type:SP_time:20250211_223029_dim8_batch64_lr0.0008_epoch5000_-augmyaug_setlen2/epoch_73.pt"
            model = MLP_module_8_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==16:
            model_path = CKPT_FOLDER/f"mlpckpt/type:SP_time:20250211_223219_dim16_batch64_lr0.0008_epoch5000_-augmyaug_setlen2/epoch_76.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="all_auglg":
        if dim==4:
            model_path = CKPT_FOLDER/f"mlpckpt/type:SP_time:20250212_004332_dim4_batch64_lr0.0008_epoch5000_-auglg_setlen3/epoch_51.pt"
            model = MLP_module_4_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==8:
            model_path = CKPT_FOLDER/f"/mlpckpt/type:SP_time:20250212_005413_dim8_batch64_lr0.0008_epoch5000_-auglg_setlen3/epoch_35.pt"
            model = MLP_module_8_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==16:
            model_path = CKPT_FOLDER/f"mlpckpt/type:SP_time:20250211_223219_dim16_batch64_lr0.0008_epoch5000_-augmyaug_setlen2/epoch_76.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    
    if dataset=="pairs_7scenes_stairs":
        if dim==16:
            print("pairs_7scenes_stairs loaded!!")
            model_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train/sparse/mlp/ckpts/20250507_112032/epoch_115.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
            # breakpoint()
    if dataset=="pairs_7scenes_heads":
        if dim==16:
            print("pairs_7scenes_heads loaded!!")
            model_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_heads/train/sparse/mlp/ckpts/20250507_160202/epoch_171.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="match_pos_neg_7scenes_stairs":
        if dim==16:
            print("match_pos_neg_7scenes_stairs loaded!!")
            model_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train/sparse/mlp2/ckpts/20250521_113506/epoch_212.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    
    return model


def get_mlp_augment(dim=16, dataset=None):
    CKPT_FOLDER = Path(f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc")
    scene_name = dataset[8:]
    if scene_name=="pgt_7scenes_stairs":
        if dim==16:
            model_path = CKPT_FOLDER/f"7_scenes/{scene_name}/mlpckpt/type:SP_time:20250207_192934_dim16_batch64_lr0.0008_epoch4000_descr640_SP-k1024-nms4-7_scenes_pgt_7scenes_stairs-auglg_setlen20/epoch_112.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    return model


def get_mlp_data_7scenes_Cambridege(dim=16, dataset="dataset_7scenes"):
    CKPT_FOLDER = Path(f"/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc")
    if dataset=="dataset_7scenes":
        if dim==8:
            model_path = CKPT_FOLDER/f"7_scenes/mlpckpt/7_scenes_type:SP_time:20250215_230700_dim8_batch64_lr0.0008_epoch4000_logNone/epoch_96.pt"
            model = MLP_module_8_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==16:
            model_path = CKPT_FOLDER/f"7_scenes/mlpckpt/7_scenes_type:SP_time:20250215_225922_dim16_batch64_lr0.0008_epoch4000_logNone/epoch_99.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    if dataset=="dataset_Cambridge":
        if dim==8:
            model_path = CKPT_FOLDER/f"Cambridge/mlpckpt/Cambridge_type:SP_time:20250215_232032_dim8_batch64_lr0.0008_epoch4000_logNone/epoch_424.pt"
            model = MLP_module_8_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
        if dim==16:
            model_path = CKPT_FOLDER/f"Cambridge/mlpckpt/Cambridge_type:SP_time:20250215_231750_dim16_batch64_lr0.0008_epoch4000_logNone/epoch_819.pt"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
    return model


# python -m encoders.superpoint.mlp
if __name__=="__main__":
    model = get_mlp_augment(dim=16, dataset="augment_pgt_7scenes_stairs")
    breakpoint()
