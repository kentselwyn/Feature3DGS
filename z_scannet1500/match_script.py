import os
import argparse
from z_scannet1500.utils.match_utils import match_eval


def run_ImgMatch(args):
    all_path = "/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test"
    folders = os.listdir(all_path)
    folders = sorted(folders, key=lambda f: int(f[5:9]))
    folders = [os.path.join(all_path, f, 'sfm_sample') for f in folders]
    feature_name = f"{args.method}_imrate:{args.resize_num}_th:{args.th}_mlpdim:{args.mlp_dim}_"+\
                    f"kptnum:{int(args.max_num_keypoints)}_score{args.score_scale}"
    args.feature_name = feature_name
    if args.resize_num == 1:
        args.image_folder = "images"
    else:
        args.image_folder = f"images_s{args.resize_num}"
    for fold_path in folders:
        print(f"processing {fold_path}...")
        args.input = fold_path
        _ = match_eval(args)

# python -m z_scannet1500.match_script --mlp_dim 8  --method SP --resize_num 1
# python -m z_scannet1500.match_script --mlp_dim 16 --method SP --resize_num 1
# python -m z_scannet1500.match_script --mlp_dim 8  --method SP_scannet --resize_num 2
# python -m z_scannet1500.match_script --mlp_dim 16 --method SP --resize_num 2
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resize_num", default=1,)
    parser.add_argument("--mlp_dim", type=int, default=16,)
    parser.add_argument("--th", type=float, default=0.01,)
    parser.add_argument("--max_num_keypoints", type=float, default=1024,)
    parser.add_argument("--method", required=True,)
    parser.add_argument("--kernel_size",default=15)
    parser.add_argument("--score_kpt_th", default=None)
    parser.add_argument("--histogram_th", default=0.9)
    parser.add_argument("--score_scale", default=0.6)
    parser.add_argument('--save_img', action='store_true', help='Save the image')
    args = parser.parse_args()
    args.match_name = f"match_result_KS{args.kernel_size}_SH{args.histogram_th}_SCs{args.score_scale}"
    run_ImgMatch(args)
