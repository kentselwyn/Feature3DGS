# bash zenith_scripts/train.sh
SOURSE_PATH=$1
out_name=$2
score_loss=$3
img_name=$4
feature_name="features/$5"
score_scale=$6

OUT_PATH="$SOURSE_PATH/outputs/$out_name"

python train.py -s "$SOURSE_PATH" -m "$OUT_PATH" -i "$img_name" -f "$feature_name" --iterations 8000 --score_loss "$score_loss" --score_scale $score_scale

cp "$0" "$OUT_PATH"