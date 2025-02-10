# bash zenith_scripts/train.sh
# SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test/scene0757_00/sfm_sample"
SOURSE_PATH=$1

score_loss=L2
img_name=$3
feature_name="features/$2"
out_name="$2"
score_scale=$4



OUT_PATH="$SOURSE_PATH/outputs/$out_name"

python train.py -s "$SOURSE_PATH" -m "$OUT_PATH" -i "$img_name" -f "$feature_name" --iterations 8000 --score_loss "$score_loss" --score_scale $score_scale

cp "$0" "$OUT_PATH"