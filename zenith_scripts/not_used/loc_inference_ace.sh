
data_name=$1
scene_name=$2
mlp_method=$3
mlp_dim=$4
iteration=$5
save_match=$6
lg_th=$7
kernel_size=$8
kpt_hist=$9
sp_th=$10

ace_ckpt="/home/koki/code/cc/ace/ace_models/7Scenes_pgt/$scene_name.pt"

export MLP_DIM=$mlp_dim
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
out_name="${mlp_method}_imrate:1_th:0.01_mlpdim:${mlp_dim}_kptnum:1024_score0.6_rgb"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"
if [ "$save_match" ]; then
    python loc_inference_ace.py -m $OUT_PATH --ace_ckpt $ace_ckpt --iteration $iteration \
                                --mlp_dim $mlp_dim --mlp_method $mlp_method --save_match --lg_th $lg_th \
                                --kernel_size $kernel_size --sp_th $sp_th
else
    python loc_inference_ace.py -m $OUT_PATH --ace_ckpt $ace_ckpt --iteration $iteration \
                                --mlp_dim $mlp_dim --mlp_method $mlp_method --lg_th $lg_th \
                                --kernel_size $kernel_size --sp_th $sp_th
fi

# Commands
# bash zenith_scripts/loc_inference_ace.sh Cambridge Cambridge_StMarysChurch Cambridge 32 30000 true 0.01 9 0.9 0.015
# bash zenith_scripts/loc_inference_ace.sh 7_scenes pgt_7scenes_stairs pgt_7scenes_stairs 16 30000 true 0.015 9 0.9 0.02

# 工作:
# 1. 圖片帶rotation error, translation error
# 2. 最後數據寫在一起
# 3. 紀錄各個rotation, translation
