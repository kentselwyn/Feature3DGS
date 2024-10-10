import torch
from torch import nn
from pathlib import Path
from torchrl.modules import MLP




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



CKPT_FOLDER = Path("/home/koki/gluetrain/data/ckpt/mlp/")

def get_mlp_model(dim = 16, type = "short"):
    if type=="short":
        if dim==16:
            model_path = CKPT_FOLDER/"short_pair_16/model_20240221_105233_496"
            model = MLP_module_16_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
            return model
        if dim==8:
            model_path = CKPT_FOLDER/"short_pair_8/model_20240221_105112_498"
            model = MLP_module_8_short()
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
            return model



