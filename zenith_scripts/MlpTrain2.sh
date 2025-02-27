
data_name=$1
mlp_dim=$2

log_name="log_${mlp_dim}_${data_name}_augID"
log_path="/home/koki/code/cc/feature_3dgs_2/${log_name}.txt"

desc_names=(
    "Cambridge_KingsCollege/desc_data/r1024_SP-k1024-nms4-Cambridge_Cambridge_KingsCollege-augidentity_setlen1.h5"
    "Cambridge_OldHospital/desc_data/r1024_SP-k1024-nms4-Cambridge_Cambridge_OldHospital-augidentity_setlen1.h5"
    "Cambridge_ShopFacade/desc_data/r1024_SP-k1024-nms4-Cambridge_Cambridge_ShopFacade-augidentity_setlen4.h5"
    "Cambridge_StMarysChurch/desc_data/r1024_SP-k1024-nms4-Cambridge_Cambridge_StMarysChurch-augidentity_setlen1.h5"
)


nohup python -u -m mlp.train --dim $mlp_dim --data_name $data_name \
                                --desc_names "${desc_names[@]}" --num_workers 0 \
                                --log_name $log_name \
                                > $log_path 2>&1 &

# bash zenith_scripts/MlpTrain2.sh Cambridge 16
# bash zenith_scripts/MlpTrain2.sh Cambridge 8
