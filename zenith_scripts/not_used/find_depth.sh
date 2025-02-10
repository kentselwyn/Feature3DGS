SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_chess"
name="SP_7scenes_chess_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_rgb"

OUT_PATH="$SOURSE_PATH/outputs/$name"

python find_depth.py -m $OUT_PATH



# bash zenith_scripts/find_depth.sh
