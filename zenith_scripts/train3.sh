# bash zenith_scripts/train3.sh
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/all_data/scene0724_00/A"

img_name="all_images/img648x484_raw"
feature_name="features/img648x484_feature"
out_name="12"

OUT_PATH="$SOURSE_PATH/outputs/$out_name"

python train.py -s "$SOURSE_PATH" -m "$OUT_PATH" -i "$img_name" -f "$feature_name" --iterations 10000 --eval --score_loss L2

cp "$0" "$OUT_PATH"