from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation
import torch
import cv2
from PIL import Image
import numpy as np
import requests
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision.utils import save_image


if __name__=="__main__":
    # img_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_pumpkin/test/rgb/seq-01-frame-000033.color.png"
    img_path = "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs/test/rgb/seq-01-frame-000139.color.png"
    # img_path = "/home/koki/code/cc/feature_3dgs_2_copy/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs/test/rgb/seq-01-frame-000201.color.png"

    bgr_image = cv2.imread(img_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_image)


    processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
    model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")
    model.eval()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.base_model(**inputs).last_hidden_state  # shape: [1, C, H, W]

    feature_map = features[0]  # shape: [C, H, W]
    C, H, W = feature_map.shape
    flattened = feature_map.permute(1, 2, 0).reshape(-1, C).cpu().numpy()

    pca = PCA(n_components=3)
    rgb_feat = pca.fit_transform(flattened)
    rgb_feat = rgb_feat.reshape(H, W, 3)
    rgb_feat -= rgb_feat.min()
    rgb_feat /= rgb_feat.max()

    # breakpoint()
    raw_img = inputs['pixel_values'][0]
    

    plt.imshow(rgb_feat)
    plt.title("Feature Map Visualized")
    plt.axis("off")
    out_name = Path(img_path).stem

    save_image(raw_img, f"{out_name}.png")
    plt.savefig(f"{out_name}_feat.png")
    # breakpoint()

# python mobileViT_vis.py
