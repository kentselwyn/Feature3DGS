
data_name=$1
scene_name=$2
mlp_method=$3
mlp_dim=$4
view_num=$5

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
out_name="${mlp_method}_imrate:1_th:0.01_mlpdim:${mlp_dim}_kptnum:1024_score0.6_rgb"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"
python render.py -m $OUT_PATH --iteration 15000 --skip_train --view_num $view_num

# bash zenith_scripts/render.sh Cambridge Cambridge_KingsCollege Cambridge 32
# bash zenith_scripts/render.sh 7_scenes pgt_7scenes_stairs augment_pgt_7scenes_stairs 16 200
