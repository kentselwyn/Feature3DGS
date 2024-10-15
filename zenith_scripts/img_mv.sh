# bash zenith_scripts/img_mv.sh

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/all_data/scene0708_00/A"
IMAGE="img1296x968_raw"
kpt_name="img1296x968_feature"

python img_mv.py -s "$SOURSE_PATH" -i "$IMAGE" -k "$kpt_name"