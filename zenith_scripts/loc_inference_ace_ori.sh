# Commands
# ( bash zenith_scripts/loc_inference_ace_ori.sh )
###############################################
# 7_scenes
# Cambridge
data_name="7_scenes"
###############################################
# pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_pumpkin, pgt_7scenes_redkitchen, pgt_7scenes_stairs
# Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
scene_name="pgt_7scenes_chess" 
# scene_name=$1
###############################################
save_match=0
iteration=30000
sp_th=0.01
lg_th=0.01
ransac_iters=20000
stop_kpt_num=50
pnp_option="pycolmap" #iters, epnp, pycolmap
depth_render=1
rival=1

###############################################
export DEPTH_RENSER=$depth_render

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
name="raw"
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
    test_name="iteration${iteration}_sp${sp_th}_lg${lg_th}_"
    test_name+="ransaciters${ransac_iters}_"
    test_name+="stop_kpt_num${stop_kpt_num}_pnp_${pnp_option}_ace_DepthRender"
else
    test_name="iteration${iteration}_sp${sp_th}_lg${lg_th}_"
    test_name+="ransaciters${ransac_iters}_"
    test_name+="stop_kpt_num${stop_kpt_num}_pnp_${pnp_option}_ace"
fi

if (( rival )); then
    test_name+="_rival_000"
fi

###############################################
if (( save_match )); then
    python loc_inference_ace_ori.py --save_match \
                                -m $OUT_PATH --ace_ckpt $ace_ckpt --iteration $iteration \
                                --lg_th $lg_th \
                                --sp_th $sp_th --ransac_iters $ransac_iters \
                                --stop_kpt_num $stop_kpt_num --pnp $pnp_option \
                                --test_name "$test_name" --rival "$rival"
else
    python loc_inference_ace_ori.py -m $OUT_PATH --ace_ckpt $ace_ckpt --iteration $iteration \
                                --lg_th $lg_th \
                                --sp_th $sp_th --ransac_iters $ransac_iters \
                                --stop_kpt_num $stop_kpt_num --pnp $pnp_option \
                                --test_name "$test_name" --rival "$rival"
fi
