# bash zenith_scripts/match_blending.sh

type="drjohnson"
# out_name="SP_tank_db_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_images_low_resolution"
out_name="SP_tank_db_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6"

OUT_PATH="/home/koki/code/cc/feature_3dgs_2/img_match/Else/tandt_db/$type/outputs/$out_name"

python match_blending_truck.py -m $OUT_PATH
