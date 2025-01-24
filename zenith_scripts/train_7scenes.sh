SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_office"
name="SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_rgb"
feature_name="features/$name"
OUT_PATH="$SOURSE_PATH/outputs/$name"
score_loss=L2


python train.py -s "$SOURSE_PATH" -m "$OUT_PATH" -f "$feature_name" --iterations 30000 --score_loss "$score_loss" --score_scale 0.6


# bash zenith_scripts/train_7scenes.sh


# export MLP dataset -> train MLP(use b5,b6) -> use MLP build gaussian feature set(tarin+test) 
# -> train gaussian -> localization inference
