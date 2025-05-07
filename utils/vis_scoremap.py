import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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

def overlay_scoremap(image, scoremap):
    # Normalize the scoremap to the range [0, 1]
    scoremap = (scoremap - scoremap.min()) / (scoremap.max() - scoremap.min())
    scoremap = (scoremap * 255).astype(np.uint8)

    # Convert the scoremap to a PIL image
    scoremap_image = Image.fromarray(scoremap)
    image = Image.fromarray(image)
    # Resize the scoremap to match the original image size
    scoremap_image = scoremap_image.resize(image.size, Image.BILINEAR)

    # Overlay the scoremap on the original image
    overlayed_image = Image.blend(image.convert("RGBA"), scoremap_image.convert("RGBA"), alpha=0.5)

    # Draw circles on high-value coordinates
    draw = ImageDraw.Draw(overlayed_image)
    high_value_coords = np.argwhere(scoremap == 255)  # Find coordinates of high values (255 after normalization)
    scale_x = image.size[0] / scoremap.shape[1]
    scale_y = image.size[1] / scoremap.shape[0]

    for y, x in high_value_coords:
        # Scale coordinates to match the resized scoremap
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        radius = 3  # Radius of the circle
        draw.ellipse(
            [(scaled_x - radius, scaled_y - radius), (scaled_x + radius, scaled_y + radius)],
            outline="red",
            width=2
        )

    return overlayed_image


# python -m codes.vis_scoremap
if __name__=="__main__":
    path = Path("/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_01/B/features/sp_feature_2/5366_smap_CxHxW.pt")
    smap = torch.load(path)
    print(smap.shape)

    name = path.stem

    d_smap = one_channel_vis(smap)
    d_smap.save(f"{name}.png")




