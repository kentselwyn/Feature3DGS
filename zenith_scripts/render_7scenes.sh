SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_chess"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs"
name="SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_rgb"
# "SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_rgb"
# "SP_7scenes_chess_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_rgb"
OUT_PATH="$SOURSE_PATH/outputs/$name"

python render.py -m $OUT_PATH --iteration 30000 --skip_train

cp "$0" "$OUT_PATH"

# bash zenith_scripts/render_7scenes.sh
# export MLP dataset -> train MLP(use b5,b6) -> use MLP build gaussian feature set(tarin+test) -> train gaussian -> localization inference