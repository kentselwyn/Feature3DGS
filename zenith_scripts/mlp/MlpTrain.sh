
data_name=$1
mlp_dim=$2

log_name="log_${mlp_dim}_${data_name}_augID"
log_path="/home/koki/code/cc/feature_3dgs_2/${log_name}.txt"

desc_names=(
    "pgt_7scenes_chess/desc_data/r640_SP-k1024-nms4-7_scenes_pgt_7scenes_chess-augidentity_setlen2.h5"
    "pgt_7scenes_fire/desc_data/r640_SP-k1024-nms4-7_scenes_pgt_7scenes_fire-augidentity_setlen3.h5"
    "pgt_7scenes_heads/desc_data/r640_SP-k1024-nms4-7_scenes_pgt_7scenes_heads-augidentity_setlen5.h5"
    "pgt_7scenes_office/desc_data/r640_SP-k1024-nms4-7_scenes_pgt_7scenes_office-augidentity_setlen1.h5"
    "pgt_7scenes_pumpkin/desc_data/r640_SP-k1024-nms4-7_scenes_pgt_7scenes_pumpkin-augidentity_setlen2.h5"
    "pgt_7scenes_redkitchen/desc_data/r640_SP-k1024-nms4-7_scenes_pgt_7scenes_redkitchen-augidentity_setlen1.h5"
    "pgt_7scenes_stairs/desc_data/r1024_SP-k1024-nms4-7_scenes_pgt_7scenes_stairs-augidentity_setlen3.h5"
)
# scene_names+=(
#     "Cambridge_KingsCollege" "Cambridge_OldHospital" "Cambridge_ShopFacade" "Cambridge_StMarysChurch"
# )


nohup python -u -m mlp.train --dim $mlp_dim --data_name $data_name \
                                --desc_names "${desc_names[@]}" --num_workers 0\
                                > $log_path 2>&1 &
# python -u -m mlp.train --dim $mlp_dim --data_name $data_name \
#                                 --desc_names "${desc_names[@]}" --num_workers 0 \
#                                 --log_name $log_name
# bash zenith_scripts/MlpTrain.sh 7_scenes 16
# bash zenith_scripts/MlpTrain.sh 7_scenes 8
