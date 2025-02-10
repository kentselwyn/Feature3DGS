# bash zenith_scripts/render_single.sh


OUT_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs/outputs/SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_rgb"


python render.py -m $OUT_PATH --iteration 30000 --skip_train --view_num 20
cp "$0" "$OUT_PATH"
