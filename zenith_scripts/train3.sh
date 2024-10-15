# bash zenith_scripts/train3.sh
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/all_data/scene0724_00/A"

img_name="img1296x968_raw"
feature_name="img1296x968_feature"
out_name="5"

OUT_PATH="$SOURSE_PATH/outputs/$out_name"

python train.py -s "$SOURSE_PATH" -m "$OUT_PATH" -i "$img_name" -f "$feature_name" --iterations 10000 --eval

cp "$0" "$OUT_PATH"