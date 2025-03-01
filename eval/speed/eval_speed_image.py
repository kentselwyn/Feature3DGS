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
from matchers.lightglue import LightGlue


LG_THRESHOLD = 0.01
KERNEL_SIZE = 7
SCORE_KPT_THRESHOLD = 0.02
MLP_DIM = 8
IM_RATE = 1

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


# raw_imrate:1
# SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_score0.6
# SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6
# SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_score0.6
# SP_scannet_imrate:2_th:0.01_mlpdim:8_kptnum:1024_score0.6
# NAME = "SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_score0.6"
NAME = f"SP_imrate:{IM_RATE}_th:0.01_mlpdim:{MLP_DIM}_kptnum:1024_score0.6"


over_all_result = f'match_speed_{NAME}.txt'
if os.path.exists(over_all_result):
    os.remove(over_all_result)


scene_path = "/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test"
folders = os.listdir(scene_path)
folders = sorted(folders)


indices = [8, 12, 20, 27, 55]
folders = [folders[i] for i in indices]
folder_paths = [os.path.join(scene_path, f, f"sfm_sample/outputs/{NAME}") for f in folders]
print(folders)


def get_speed():
    pipe_param = PipelineParams()
    view_num = 0
    elapsed = 0

    for idx, model_path in enumerate(folder_paths):
        print(f"processing {model_path}......")
        cfgfile_string = "Namespace()"
        args = get_arg_dict(cfgfile_string, model_path)
        args.eval = False
        args.score_kpt_th = SCORE_KPT_THRESHOLD
        args.kernel_size = KERNEL_SIZE
        args.histogram_th = None
        args.mlp_dim = MLP_DIM
        args.method = "SP"

        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, shuffle=False, load_iteration=8000)
        cams = scene.getTrainCameras()


        bg_color = [1,1,1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        data = {}
        random.seed(0)
        cam_len = len(cams)
        random_int = [random.randint(0, cam_len-1) for _ in range(cam_len)]
        

        test_cams = cams[:20]
        for j in range(1):
            for idx, view0 in enumerate(test_cams):
                view1 = test_cams[random_int[idx]]
                r_pkg0 = render(view0, gaussians, pipe_param, background)
                r_pkg1 = render(view1, gaussians, pipe_param, background)
                f0 = r_pkg0["feature_map"]
                f1 = r_pkg1['feature_map']
                f0 = F.interpolate(f0.unsqueeze(0), size=(121, 162), mode='bilinear', align_corners=True).squeeze(0)
                f1 = F.interpolate(f1.unsqueeze(0), size=(121, 162), mode='bilinear', align_corners=True).squeeze(0)
                data['s0'] = r_pkg0['score_map']
                data['ft0'] = f0
                data['s1'] = r_pkg1['score_map']
                data['ft1'] = f1

                score_feature_match(data, args, matcher)
                torch.cuda.empty_cache()
        
        start = time.time()
        for j in range(2):
            print(j)
            view_num += cam_len
            for idx, view0 in enumerate(cams):
                view1 = cams[random_int[idx]]
                r_pkg0 = render(view0, gaussians, pipe_param, background)
                r_pkg1 = render(view1, gaussians, pipe_param, background)
                f0 = r_pkg0["feature_map"]
                f1 = r_pkg1['feature_map']
                f0 = F.interpolate(f0.unsqueeze(0), size=(121, 162), mode='bilinear', align_corners=True).squeeze(0)
                f1 = F.interpolate(f1.unsqueeze(0), size=(121, 162), mode='bilinear', align_corners=True).squeeze(0)
                data['s0'] = r_pkg0['score_map']
                data['ft0'] = f0
                data['s1'] = r_pkg1['score_map']
                data['ft1'] = f1

                score_feature_match(data, args, matcher)
                torch.cuda.empty_cache()

        end = time.time()
        elapsed += end-start
        torch.cuda.empty_cache()
    
    print(f'{NAME} speed')
    print('elapsed time: ', elapsed)
    print(f'view num: {view_num}')
    print(f'fps: {view_num/elapsed}')
    print()
    with open(over_all_result, 'a') as file:
        file.write(f'{NAME} speed\n')
        file.write(f'elapsed time: {elapsed}\n')
        file.write(f'view num: {view_num}\n')
        file.write(f'fps: {view_num/elapsed}\n')
        file.write('\n\n')








# python eval_speed2.py
if __name__=="__main__":
    get_speed()


