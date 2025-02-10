
data_name=$1
scene_name=$2
mlp_dim=$3
desc_name="r640_SP-k1024-nms4-7_scenes_pgt_7scenes_stairs-auglg_setlen20"
log_path="/home/koki/code/cc/feature_3dgs_2/${desc_name}.txt"

nohup python -u -m mlp.train --dim $mlp_dim --data_name $data_name \
                                --scene_name $scene_name  --desc_name $desc_name --num_workers 0\
                                > $log_path 2>&1 &

# bash zenith_scripts/mlp_train.sh 7_scenes pgt_7scenes_stairs 16
