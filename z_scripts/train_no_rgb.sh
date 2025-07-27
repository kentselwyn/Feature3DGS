start_train=0
start_render=0
start_loc=1

# Define all 7Scenes dataset scenes
scenes=("chess" "redkitchen" "stairs")

# Common parameters for all scenes
{
    ################### training ####################
        data_name="7scenes"
        iterations=60000
        score_loss="weighted" # L2, weighted, L1   
        score_scale=0.6
    ################### feature ####################
        num_kpts=512
        detect_th=0.005
        mlp_dim=16
    ###################  rendering   ###################
        render_num=100
        sp_kpt=1024
        sp_th=0.0
        hist=0.95
        match_precision_th="5e-5"
        ###############
        render_train=0
        render_test=1
        ###############
        render_kpt_desc=0
        render_match=0
    ################### localization ###################
        match_type=6
        test_iteration=$iterations
        ###############
        loc_sp_th=0.0
        loc_lg_th=0.01
        score_hist=0.95
        max_num_kpt=1024
        kernel_size=15
        ###############
        stop_kpt_num=50
        pnp_option="pycolmap" #iters, epnp, pycolmap
    ####################################################
        save_match=1
        save_render_img=0
}

# Iterate through all scenes
for scene in "${scenes[@]}"; do
    echo "========================================"
    echo "Processing scene: $scene"
    echo "========================================"
    
    # Set scene-specific parameters
    scene_name="pgt_7scenes_${scene}"
    mlp_name="pgt_7scenes_${scene}"
    ace_ckpt="/work/u8351896/ACE/$scene_name.pt"
    
    # Set paths
    # out_name="1_${iterations}_${score_loss}_${score_scale}_${num_kpts}_${detect_th}_${mlp_dim}_${mlp_name}"
    out_name="no_rgb"
    test_name="${test_iteration}_${loc_sp_th}_${loc_lg_th}_"
    test_name+="${score_hist}_${max_num_kpt}_${kernel_size}_${stop_kpt_num}_${pnp_option}"
    
    SOURSE_PATH="/work/u8351896/$data_name/$scene_name"
    OUT_PATH="/work/u8351896/$data_name/$scene_name/outputs/$out_name"
    
    # Training
    if (( start_train )); then
        echo "Training for scene: $scene"
        python train_no_rgb.py -s "$SOURSE_PATH" -i "rgb" -m "$OUT_PATH" --iterations $iterations \
                        --score_loss "$score_loss" --score_scale "$score_scale"  \
                        --num_kpts $num_kpts --detect_th $detect_th --mlp_dim $mlp_dim --mlp_name $mlp_name --use_abs_grad --load_testcam --features_only
    fi

    # Rendering
    if (( start_render )); then
        echo "Rendering for scene: $scene"
        detector_path="data/detector/7_scenes/pgt_7scenes_${scene}/L2_0.0001_sptrain_normalizeRemoveNeg/epoch_138.pt"
        python render.py -m $OUT_PATH --iteration $iterations --view_num $render_num \
                            --sp_kpt $sp_kpt --sp_th $sp_th --detector_path $detector_path --hist $hist\
                            --match_precision_th $match_precision_th\
                            $([[ "$render_train" -eq 1 ]] && echo "--render_train") \
                            $([[ "$render_test" -eq 1 ]] && echo "--render_test") \
                            $([[ "$render_kpt_desc" -eq 1 ]] && echo "--render_kpt_desc") \
                            $([[ "$render_match" -eq 1 ]] && echo "--render_match")
    fi

    # Localization
    if (( start_loc )); then
        echo "Localization for scene: $scene"
        python -m z_localization.loc_inference_with_plots -m $OUT_PATH --iteration $test_iteration \
                                                --mlp_dim $mlp_dim --mlp_method $mlp_name \
                                                --sp_th $loc_sp_th --lg_th $loc_lg_th --kpt_hist $score_hist \
                                                --ransac_iters 20000 --kernel_size $kernel_size \
                                                $([[ "$save_match" -eq 1 ]] && echo "--save_match")
    fi
    
    echo "Completed processing scene: $scene"
    echo ""
done

echo "========================================"
echo "All scenes processed successfully!"
echo "========================================"