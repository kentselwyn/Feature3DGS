# Commands
# ( bash zenith_scripts/loc_inference_ace_ori_gscpr.sh )
{
    ###############################################
    # 7_scenes
    # Cambridge
    data_name="7_scenes"
    ###############################################
    # chess, fire, heads, office, pumpkin, redkitchen, stairs
    # Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
    scene_name="${1}" 
    # scene_name=$1
    ###############################################
    save_match=0
    iteration=30000
    sp_th=0.01
    lg_th=0.01
    ransac_iters=2000
    stop_kpt_num=50
    pnp_option="pycolmap" #iters, epnp, pycolmap
    depth_render=$2
    # 0:sp,lg, 1:mast3r, 2:aspan
    match_type=0
}

###############################################
export DEPTH_RENSER=$depth_render
{
              # "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train"
    SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/$data_name/scene_$scene_name/train"
    name="raw"
    OUT_PATH="$SOURSE_PATH/outputs/$name"

    ###############################################
    if [[ "$data_name" == "7_scenes" ]]; then
        ace_ckpt="/home/koki/code/cc/feature_3dgs_2/data/ace_models/7Scenes_pgt/pgt_7scenes_${scene_name}.pt"
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

    if [ "$match_type" -eq 0 ]; then
        test_name+="_rival_000"
    fi

    if [ "$match_type" -eq 1 ]; then
        test_name+="_mast3r"
    fi

    if [ "$match_type" -eq 2 ]; then
        test_name+="_aspan"
    fi
}
###############################################
{
    if (( save_match )); then
        python -m z_localization.loc_inference_ace_ori --save_match \
                                    -m $OUT_PATH --ace_ckpt $ace_ckpt --iteration $iteration \
                                    --lg_th $lg_th \
                                    --sp_th $sp_th --ransac_iters $ransac_iters \
                                    --stop_kpt_num $stop_kpt_num --pnp $pnp_option \
                                    --test_name "$test_name" --match_type "$match_type"
    else
        python -m z_localization.loc_inference_ace_ori \
                                    -m $OUT_PATH --ace_ckpt $ace_ckpt --iteration $iteration \
                                    --lg_th $lg_th \
                                    --sp_th $sp_th --ransac_iters $ransac_iters \
                                    --stop_kpt_num $stop_kpt_num --pnp $pnp_option \
                                    --test_name "$test_name" --match_type "$match_type"
    fi
}
