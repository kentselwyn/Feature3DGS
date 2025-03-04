# Commands
# ( bash zenith_scripts/loc_inference_ace.sh )
###############################################
# 7_scenes
# Cambridge
data_name="7_scenes"
###############################################
# pgt_7scenes_pumpkin, pgt_7scenes_stairs, pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_redkitchen
# Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
scene_name="pgt_7scenes_stairs" 
###############################################
# all
# pgt_7scenes_pumpkin, augment_pgt_7scenes_stairs, pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_redkitche
# Cambridge
# Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
mlp_method="augment_pgt_7scenes_stairs"
###############################################
mlp_dim=16
save_match=1
iteration=30000
sp_th=0.02
lg_th=0.01
kpt_hist=0.9
ransac_iters=20000
kernel_size=13
stop_kpt_num=50
pnp_option="pycolmap" #iters, epnp, pycolmap
depth_render=1
max_num_keypoints=512
score_loss="L2"
score_scale=0.6
resize_num=1
rival=0

###############################################
export MLP_DIM=$mlp_dim
export DEPTH_RENSER=$depth_render

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
name="${mlp_method}_imrate:${resize_num}_th:${sp_th}_"
name+="mlpdim:${mlp_dim}_kptnum:${max_num_keypoints}_"
name+="Score${score_loss}_ScoreScale${score_scale}_rgb"
OUT_PATH="$SOURSE_PATH/outputs/$name"

###############################################
if [[ "$data_name" == "7_scenes" ]]; then
    ace_ckpt="/home/koki/code/cc/feature_3dgs_2/data/ace_models/7Scenes_pgt/$scene_name.pt"
elif [[ "$data_name" == "Cambridge" ]]; then
    ace_ckpt="/home/koki/code/cc/feature_3dgs_2/data/ace_models/Cambridge/$scene_name.pt"
else
    echo "Dataset not exist"
fi

###############################################
if (( depth_render )); then
    test_name="iteration${iteration}_sp${sp_th}_lg${lg_th}_kptth0.01_"
    test_name+="kpthist${kpt_hist}_ransaciters${ransac_iters}_KptKernalSize${kernel_size}_"
    test_name+="stop_kpt_num${stop_kpt_num}_pnp_${pnp_option}_ace_DepthRender"
else
    test_name="iteration${iteration}_sp${sp_th}_lg${lg_th}_kptth0.01_"
    test_name+="kpthist${kpt_hist}_ransaciters${ransac_iters}_KptKernalSize${kernel_size}_"
    test_name+="stop_kpt_num${stop_kpt_num}_pnp_${pnp_option}_ace"
fi

if (( rival )); then
    test_name+="_rival_000"
fi

###############################################
if (( save_match )); then
    python loc_inference_ace.py --save_match \
                                -m $OUT_PATH --ace_ckpt $ace_ckpt --iteration $iteration \
                                --mlp_dim $mlp_dim --mlp_method $mlp_method --lg_th $lg_th \
                                --kernel_size $kernel_size --sp_th $sp_th --ransac_iters $ransac_iters \
                                --stop_kpt_num $stop_kpt_num --pnp $pnp_option --kpt_hist $kpt_hist \
                                --test_name "$test_name" --rival "$rival"
else
    python loc_inference_ace.py -m $OUT_PATH --ace_ckpt $ace_ckpt --iteration $iteration \
                                --mlp_dim $mlp_dim --mlp_method $mlp_method --lg_th $lg_th \
                                --kernel_size $kernel_size --sp_th $sp_th --ransac_iters $ransac_iters \
                                --stop_kpt_num $stop_kpt_num --pnp $pnp_option --kpt_hist $kpt_hist \
                                --test_name "$test_name" --rival "$rival"
fi
