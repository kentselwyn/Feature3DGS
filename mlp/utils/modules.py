import torch
from torch import nn
from pathlib import Path
from torchrl.modules import MLP
import mlp.utils as utils




class MLP_module_4_long(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=4   , num_cells=[128, 64, 32, 32, 16, 16, 8, 4])
        self.MLP_de = MLP(in_features=4, out_features=256,    num_cells=[4, 8, 16, 16, 32, 32, 64, 128])
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back

class MLP_module_8_long(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=8   , num_cells=[128, 64, 64, 32, 32, 16, 8])
        self.MLP_de = MLP(in_features=8, out_features=256,    num_cells=[8, 16, 32, 32, 64, 64, 128])
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back

class MLP_module_16_long(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=16   , num_cells=[256, 128, 128, 64, 64, 32, 32, 16, 16])
        self.MLP_de = MLP(in_features=16, out_features=256,    num_cells=[16, 16, 32, 32, 64, 64, 128, 128, 256])
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back





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
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back


class MLP_module_16_short(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=16   , num_cells=[128, 64, 32, 16])
        self.MLP_de = MLP(in_features=16, out_features=256,    num_cells=[16, 32, 64, 128])
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back


class MLP_module_16_short2(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=16   , num_cells=[128, 64, 32, 16])
        self.MLP_de = MLP(in_features=16, out_features=256,    num_cells=[16, 32, 64, 128])
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back


class MLP_module_16_vit(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=160, out_features=16   , num_cells=[128, 64, 32])
        self.MLP_de = MLP(in_features=16, out_features=160,    num_cells=[32, 64, 128])
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back






class MLP_module_32_short(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=32   , num_cells=[128, 64, 32])
        self.MLP_de = MLP(in_features=32,   out_features=256,  num_cells=[32, 64, 128])
        # for p in self.parameters():
        #     p.requires_grad = False
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back


class MLP_module_64_short(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=64   , num_cells=[128, 64, 64])
        self.MLP_de = MLP(in_features=64,   out_features=256,  num_cells=[64, 64, 128])
        # for p in self.parameters():
        #     p.requires_grad = False
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back


class MLP_module_128_short(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.MLP    = MLP(in_features=256, out_features=128   , num_cells=[128, 128])
        self.MLP_de = MLP(in_features=128,   out_features=256,  num_cells=[128, 128])
    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp, desc_back







class MLP_module_4_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=128, out_features=4   , num_cells=[64, 32, 16, 8])
        self.MLP_de = MLP(in_features=4,   out_features=128,  num_cells=[8, 16, 32, 64])
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back


class MLP_module_8_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=128, out_features=8   , num_cells=[64, 32, 16, 8])
        self.MLP_de = MLP(in_features=8,   out_features=128,  num_cells=[8, 16, 32, 64])
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back


class MLP_module_16_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP    = MLP(in_features=128, out_features=16   , num_cells=[64, 32, 16, 16])
        self.MLP_de = MLP(in_features=16, out_features=128,    num_cells=[16, 16, 32, 64])
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, desc: torch.Tensor):
        desc_mlp = self.MLP(desc)
        desc_back = self.MLP_de(desc_mlp)
        return desc_back





CKPT_FOLDER = Path("/home/koki/gluetrain/data/ckpt/mlp/")

def get_module_ckptpath(dim: int, type = "short"):
    if type=="short":
        if dim==4:
            model_path = CKPT_FOLDER/"short_pair_4/model_20240221_104220_497"
            return MLP_module_4_short, model_path
        elif(dim==8):
            model_path = CKPT_FOLDER/"short_pair_8/model_20240221_105112_498"
            return MLP_module_8_short, model_path
        elif(dim==16):
            model_path = CKPT_FOLDER/"short_pair_16/model_20240221_105233_496"
            return MLP_module_16_short, model_path
        elif(dim==32):
            model_path = CKPT_FOLDER/"short_pair_32/model_20240407_070737_497"
            return MLP_module_32_short, model_path
        elif(dim==64):
            model_path = CKPT_FOLDER/"short_pair_64/model_20240429_035533_496"
            return MLP_module_64_short, model_path
        elif(dim==128):
            model_path = CKPT_FOLDER/"short_pair_128"/"model_20240430_002149_10"
            return MLP_module_128_short, model_path
    elif type =="long":
        if dim==4:
            model_path = CKPT_FOLDER/"long_non_4/model_20240129_011404_496"
            return MLP_module_4_long, model_path
        elif(dim==8):
            model_path = CKPT_FOLDER/"long_non_8/model_20240129_010318_496"
            return MLP_module_8_long, model_path
        elif(dim==16):
            model_path = CKPT_FOLDER/"long_non_16/model_20240128_052453_474"
            return MLP_module_16_long, model_path
    elif type=="megadepth":
        if dim==4:
            model_path = CKPT_FOLDER/"mega4/model_20240318_060047_497"
            return MLP_module_4_short, model_path
        elif(dim==8):
            model_path = CKPT_FOLDER/"mega8/model_20240318_060116_499"
            return MLP_module_8_short, model_path
        elif(dim==16):
            model_path = CKPT_FOLDER/"mega16/model_20240318_060144_497"
            return MLP_module_16_short, model_path
    elif type=="homography":
        if dim==64:
            model_path = CKPT_FOLDER/"short_pair_64_homography/model_mlp_time20240511_214324_epoch197"
            return MLP_module_64_short, model_path



def test_model_ckpt(dim, type):
    model, ckpt_path = get_module_ckptpath(dim, type)
    model = model()
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    print(utils.count_trainable_params(model))
    print(model)
    
    return model






def get_mlp_model(dim = 16, type = "sp"):
    if type=="sp":
        name = "Superpoint"
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
    return model


def get_mlp(dim = 16):
    if dim==4:
        model = MLP_module_4_short()
    elif dim==8:
        model = MLP_module_8_short()
    elif dim==16:
        model = MLP_module_16_short()
    elif dim==32:
        model = MLP_module_32_short()
    elif dim==64:
        model = MLP_module_64_short()
    else:
        model=None
    return model

def get_mlp_128(dim = 16):
    if dim==4:
        model = MLP_module_4_128()
    elif dim==8:
        model = MLP_module_8_128()
    elif dim==16:
        model = MLP_module_16_128()
    else:
        model=None
    return model


def get_mlp_vit(dim=16):
    if dim==16:
        model = MLP_module_16_vit()
    else:
        model = None
    return model


# python -m core.MLP.modules
if __name__=="__main__":
    model = test_model_ckpt(64, "short")
    # test()







