# bash zenith_scripts/render_single.sh


OUT_PATH="/home/koki/code/cc/feature_3dgs_2/img_match/Else/tandt_db/db/playroom/outputs/SP_tank_db_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6"


python render.py -m $OUT_PATH --iteration 8000 --skip_train



cp "$0" "$OUT_PATH"
