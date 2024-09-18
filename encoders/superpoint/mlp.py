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
        desc_back = self.MLP_de(desc_mlp)
        return desc_mlp




CKPT_FOLDER = Path("/home/koki/gluetrain/data/ckpt/mlp/")

def get_module_ckptpath(dim = 16, type = "short"):
    if type=="short":
        if dim==16:
            model_path = CKPT_FOLDER/"short_pair_16/model_20240221_105233_496"
            return MLP_module_16_short, model_path



