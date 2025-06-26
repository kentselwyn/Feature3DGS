# Commands
# ( bash zenith_scripts/CPR/loc_inference.sh )
{
    ###############################################
    # 7_scenes
    # Cambridge
    data_name="7_scenes"
    ###############################################
    # chess, fire, heads, office, pumpkin, redkitchen, stairs
    # Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
    # scene_name="${1}" 
    ss="stairs"
    scene_name="scene_$ss"
    ###############################################
    save_match=1
    iteration=30000
    sp_th=0.01
    lg_th=0.01
    ransac_iters=2000
    stop_kpt_num=50
    pnp_option="pycolmap" #iters, epnp, pycolmap
    depth_render=1
    # 0:sp,lg, 1:mast3r, 2:aspan
    match_type=0
    mlp_method="match_pos_neg_7scenes_${ss}"
}

###############################################
export DEPTH_RENSER=$depth_render
{
              # "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/7_scenes/scene_stairs/train"
    SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/$data_name/$scene_name/train"
    name="match_pos_neg_7scenes_stairs_imrate:1_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6_rgb_UseTrueRender"
    OUT_PATH="$SOURSE_PATH/outputs/$name"

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
        python -m z_localization.loc_inference --save_match \
                                    --mlp_dim 16 --mlp_method $mlp_method \
                                    -m $OUT_PATH --iteration $iteration \
                                    --lg_th $lg_th \
                                    --sp_th $sp_th --ransac_iters $ransac_iters
                                    # --stop_kpt_num $stop_kpt_num --pnp $pnp_option \
                                    # --test_name "$test_name" --match_type "$match_type"
    else
        python -m z_localization.loc_inference \
                                    --mlp_dim 16 --mlp_method $mlp_method \
                                    -m $OUT_PATH --iteration $iteration \
                                    --lg_th $lg_th \
                                    --sp_th $sp_th --ransac_iters $ransac_iters
                                    # --stop_kpt_num $stop_kpt_num --pnp $pnp_option \
                                    # --test_name "$test_name" --match_type "$match_type"
    fi
}
