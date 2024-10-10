import os
from PIL import Image

input = "/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_01/A/all_images/images"
files = os.listdir(input)
files = sorted(files, key=lambda x: int(x.split('.')[0]))


out = "/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_01/A/outputs/imrate:2_th:0.01_mlpdim:16/rendering/trains/ours_7000/image_orig"


os.makedirs(f"{out}", exist_ok=True)

for i, name in enumerate(files):
    path = f"{input}/{name}"
    img = Image.open(path)
    w,h = img.size
    img = img.resize((int(w/2), int(h/2)))
    img.save(f"{out}/{i:05d}.png")

# python move_images.py
