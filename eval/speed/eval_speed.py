import os
import torch
import random
import time
from scene import Scene
from argparse import Namespace
from gaussian_renderer import render
import torch.nn.functional as F
from utils.match_img import score_feature_match
from scene.gaussian_model import GaussianModel
from encoders.superpoint.lightglue import LightGlue


# python eval_speed.py

scenes = ['0713_00']
outputs=['test_fea_score:l2_0.1']
s_len = len(scenes)


LG_THRESHOLD = 0.01
SP_THRESHOLD = 0.01
SCORE_KPT_THRESHOLD_HIGH = 0.1
SCORE_KPT_THRESHOLD_LOW = 0.05
KERNEL_SIZE_LOW = 3
KERNEL_SIZE_HIGH = 7
    

matcher = LightGlue({
            "filter_threshold": LG_THRESHOLD#0.01,
        }).to("cuda").eval()


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

NAME = f"sp:{SP_THRESHOLD}_lg:{LG_THRESHOLD}_kpt(h):{SCORE_KPT_THRESHOLD_HIGH}_kpt(l):{SCORE_KPT_THRESHOLD_LOW}_kernal(h):{KERNEL_SIZE_HIGH}_feature"
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

        gt_feature_map = cams[0].semantic_feature
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
                f0 = r_pkg0["feature_map"]
                f1 = r_pkg1['feature_map']
                f0 = F.interpolate(f0.unsqueeze(0), 
                                   size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
                f1 = F.interpolate(f1.unsqueeze(0), 
                                   size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
                data['s0'] = r_pkg0['score_map']
                data['ft0'] = f0
                data['s1'] = r_pkg1['score_map']
                data['ft1'] = f1
                en = time.time()
                # print("first: ",en-st)

                st = time.time()
                score_feature_match(data, args, matcher)
                en = time.time()
                # print("second: ",en-st)
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



        

        




