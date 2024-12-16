# bash zenith_scripts/train_single.sh
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/img_match/Else/tandt_db/truck"

score_loss=L2
img_name="images_low_resolution"

name="SP_tank_db_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_images_low_resolution"
out_name=$name
feature_name="features/$name"


OUT_PATH="$SOURSE_PATH/outputs/$out_name"

python train.py -s "$SOURSE_PATH" -m "$OUT_PATH" -i "$img_name" -f "$feature_name" --iterations 8000 --score_loss "$score_loss" --score_scale 0.6 --eval

cp "$0" "$OUT_PATH"