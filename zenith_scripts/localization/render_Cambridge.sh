
mlp_method=$1
scene_name=$2
mlp_dim=$3
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/Cambridge/$scene_name"
out_name="${mlp_method}_imrate:1_th:0.01_mlpdim:${mlp_dim}_kptnum:1024_score0.6_rgb"



python render.py -m $OUT_PATH --iteration 15000 --skip_train --view_num 200

# bash zenith_scripts/render_Cambridge.sh Cambridge Cambridge_KingsCollege 32
