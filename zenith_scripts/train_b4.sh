# bash zenith_scripts/train_b4.sh
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/all_data/scene0724_00/A"

img_name="all_images/img648x484_feature"
feature_name="features/img648x484_feature"
out_name="test_b4"

OUT_PATH="$SOURSE_PATH/outputs/$out_name"

python train_b4.py -s "$SOURSE_PATH" -m "$OUT_PATH" -i "$img_name" -f "$feature_name" --iterations 10000 --eval

cp "$0" "$OUT_PATH"