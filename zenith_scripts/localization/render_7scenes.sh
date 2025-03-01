name=$1
mlp_dim=$2
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/$name"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_chess"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs"
out_name="${name}_imrate:1_th:0.01_mlpdim:${mlp_dim}_kptnum:1024_score0.6_rgb"
# SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_rgb
# SP_7scenes_chess_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_rgb
# pgt_7scenes_stairs_imrate:1_th:0.0_mlpdim:16_kptnum:2048_score1_rgb
OUT_PATH="$SOURSE_PATH/outputs/$out_name"

python render.py -m $OUT_PATH --iteration 10000 --skip_train --view_num 200
# --view_num 200

# cp "$0" "$OUT_PATH"

# bash zenith_scripts/render_7scenes.sh pgt_7scenes_stairs 32
# export MLP dataset -> train MLP(use b5,b6) -> use MLP build gaussian feature set(tarin+test) -> train gaussian -> localization inference
