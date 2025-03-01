
data_name=$1
scene_name=$2
mlp_method=$3
mlp_dim=$4

export MLP_DIM=$mlp_dim
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
name="${mlp_method}_imrate:1_th:0.01_mlpdim:${mlp_dim}_kptnum:1024_score0.6_rgb"
feature_name="features/$name"
OUT_PATH="$SOURSE_PATH/outputs/$name"

python train.py -s "$SOURSE_PATH" -m "$OUT_PATH" -f "$feature_name" --iterations 30000 --score_loss L2 --score_scale 0.6

# bash zenith_scripts/train.sh Cambridge Cambridge_KingsCollege Cambridge 32
# bash zenith_scripts/train.sh 7_scenes pgt_7scenes_stairs augment_pgt_7scenes_stairs 16
# bash zenith_scripts/train.sh 7_scenes pgt_7scenes_stairs all 8
# bash zenith_scripts/train.sh 7_scenes pgt_7scenes_stairs all_auglg 4
