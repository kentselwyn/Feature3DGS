import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def one_channel_vis(score):
    scale_nor = score.max().item()
    score_nor = score / scale_nor
    depth_tensor_squeezed = score_nor.squeeze()  # Remove the channel dimension
    colormap = plt.get_cmap('jet')
    depth_colored = colormap(depth_tensor_squeezed.cpu().detach().numpy())
    depth_colored_rgb = depth_colored[:, :, :3]
    depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))

    return depth_image

# python -m codes.vis_scoremap
if __name__=="__main__":
    path = Path("/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_01/B/features/sp_feature_2/5366_smap_CxHxW.pt")
    smap = torch.load(path)
    print(smap.shape)

    name = path.stem

    d_smap = one_channel_vis(smap)
    d_smap.save(f"{name}.png")
