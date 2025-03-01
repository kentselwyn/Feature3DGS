
data_name=$1
scene_name=$2
mlp_method=$3
mlp_dim=$4
iteration=$5
save_match=$6

export MLP_DIM=$mlp_dim
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
out_name="${mlp_method}_imrate:1_th:0.01_mlpdim:${mlp_dim}_kptnum:1024_score0.6_rgb"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"
if [ "$save_match" ]; then
    python loc_inference.py -m $OUT_PATH --iteration $iteration --mlp_dim $mlp_dim --mlp_method $mlp_method --save_match
else
    python loc_inference.py -m $OUT_PATH --iteration $iteration --mlp_dim $mlp_dim --mlp_method $mlp_method
fi

# Commands
# bash zenith_scripts/loc_inference.sh Cambridge Cambridge_StMarysChurch Cambridge 32 30000 true
# bash zenith_scripts/loc_inference.sh 7_scenes pgt_7scenes_stairs augment_pgt_7scenes_stairs 16 30000 true
