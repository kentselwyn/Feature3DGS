import os
import torch
import time
from scene import Scene
from random import randint
from gaussian_renderer import render, GaussianModel
from encoders.utils import GroupParams, OptimizationParams, PipelineParams
import torchvision.transforms as transforms
from codes.used_codes.vis_scoremap import one_channel_vis


model_param = GroupParams()
opt_param = OptimizationParams()
pipe_param = PipelineParams()


model_param.source_path = "/home/koki/code/cc/feature_3dgs_2/all_data/scene0755_00/A"
model_param.foundation_model = "imrate:2_th:0.01_mlpdim:16"
model_param.model_path = f"{model_param.source_path}/outputs/imrate:2_th:0.01_mlpdim:16"
iteration = 7000


gaussians = GaussianModel(model_param.sh_degree)
scene = Scene(model_param, gaussians, load_iteration=iteration ,shuffle=False)


viewpoint_stack = None
bg_color = [1, 1, 1] if model_param.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


img_path = "/home/koki/code/cc/feature_3dgs_2/test_images/test/img"
s_path = "/home/koki/code/cc/feature_3dgs_2/test_images/test/score"
f_path = "/home/koki/code/cc/feature_3dgs_2/test_images/test/feature"
os.makedirs(img_path, exist_ok=True)
os.makedirs(f_path, exist_ok=True)
os.makedirs(s_path, exist_ok=True)


start = time.time()
iterations = 5000
for it in range(0, iterations):
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()

    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    render_pkg = render(viewpoint_cam, gaussians, pipe_param, background)
    img, feature_map, score_map = render_pkg["render"], render_pkg["feature_map"], render_pkg["score_map"]

    # to_pil = transforms.ToPILImage()
    # img = to_pil(img)
    # img.save(f"{path}/{it}.jpg")
    # s_map = one_channel_vis(score_map)
    # s_map.save(f"{s_path}/{it}.jpg")
    print(it)




end = time.time()
print("elapsed time: ", end-start)


# python -m encoders.rendering_speed