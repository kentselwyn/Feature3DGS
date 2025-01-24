
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_heads/train"
method="SP_7scenes_chess"
IMAGE_NMAE="rgb"

python dataset_build.py --input "$SOURSE_PATH" --mlp_dim 16 --resize_num 1 --max_num_keypoints 1024 --method "$method" --images "$IMAGE_NMAE"


# bash zenith_scripts/dataset_build.sh
