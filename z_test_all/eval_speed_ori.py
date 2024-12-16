import os
import torch
import random
import time
from scene_ori import Scene
from argparse import Namespace
from gaussian_renderer.__init__ori import render
from utils.match_img import img_match2
from scene_ori.gaussian_model import GaussianModel
from encoders.superpoint.lightglue import LightGlue
from encoders.superpoint.superpoint import SuperPoint


scenes = ['0708_00', '0713_00', '0724_00']
outputs=[2, 6]
s_len = len(scenes)



SP_THRESHOLD = 0.01
LG_THRESHOLD = 0.01
SCORE_KPT_THRESHOLD_HIGH = 0.1
SCORE_KPT_THRESHOLD_LOW = 0.05
KERNEL_SIZE_LOW = 3
KERNEL_SIZE_HIGH = 7
    

matcher = LightGlue({
            "filter_threshold": LG_THRESHOLD#0.01,
        }).to("cuda").eval()

conf = {
    "sparse_outputs": True,
    "dense_outputs": True,
    "max_num_keypoints": 1024,
    "detection_threshold": SP_THRESHOLD #0.01,
}
encoder = SuperPoint(conf).to("cuda").eval()


def get_arg_dict(cfgfile_string, model_path):
    cfgfilepath = os.path.join(model_path, "cfg_args")
    with open(cfgfilepath) as cfg_file:
        cfgfile_string = cfg_file.read()
    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    merged_dict['model_path'] = model_path
    args = Namespace(**merged_dict)

    return args


class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

NAME = f"sp:{SP_THRESHOLD}_lg:{LG_THRESHOLD}_kpt(h):{SCORE_KPT_THRESHOLD_HIGH}_kpt(l):{SCORE_KPT_THRESHOLD_LOW}_kernal(h):{KERNEL_SIZE_HIGH}_imaqge"
over_all_result = f'match_speed_{NAME}.txt'
if os.path.exists(over_all_result):
    os.remove(over_all_result)



for out in outputs:
    pipe_param = PipelineParams()
    view_num = 0
    elapsed = 0

    for i in range(s_len):
        aggregate_list = []

        scene_num = scenes[i]

        source_path = f"/home/koki/code/cc/feature_3dgs_2/all_data/scene{scene_num}/A"
        model_path = f"{source_path}/outputs/{out}"

        cfgfile_string = "Namespace()"
        args = get_arg_dict(cfgfile_string, model_path)
        args.eval = False
        args.score_kpt_th_high = SCORE_KPT_THRESHOLD_HIGH
        args.score_kpt_th_low = SCORE_KPT_THRESHOLD_LOW
        args.kernel_size_low = KERNEL_SIZE_LOW
        args.kernel_size_high = KERNEL_SIZE_HIGH

        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, shuffle=False, load_iteration=10000)
        cams = scene.getTrainCameras()

        bg_color = [1,1,1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        data = {}
        data['experi_index'] = out
        random.seed(0)
        cam_len = len(cams)
        random_int = [random.randint(0, cam_len-1) for _ in range(cam_len)]


        start = time.time()
        for j in range(3):
            print(j)
            view_num += cam_len
            for idx, view0 in enumerate(cams):
                view1 = cams[random_int[idx]]

                st = time.time()
                r_pkg0 = render(view0, gaussians, pipe_param, background)
                r_pkg1 = render(view1, gaussians, pipe_param, background)

                img0 = r_pkg0['render']
                img1 = r_pkg1['render']
                data={}
                data['img0'] = img0
                data['img1'] = img1
                img_match2(data, encoder=encoder, matcher=matcher)

                
        end = time.time()
        elapsed += end-start
    print('elapsed time: ', elapsed)
    print()
    with open(over_all_result, 'a') as file:
        file.write(f'{out} speed\n')
        file.write(f'elapsed time: {elapsed}\n')
        file.write(f'view num: {view_num}\n')
        file.write(f'fps: {view_num/elapsed}\n')
        file.write('\n\n')
                

# python eval_speed_ori.py