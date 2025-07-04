start_train=0
start_render=0
start_loc=1
{
    ################### training ####################
        data_name="7_scenes"
        scene_name="pgt_7scenes_stairs"
        iterations=30000
        score_loss="weighted" # L2, weighted, L1   
        score_scale=0.6
    ################### feature ####################
        num_kpts=512
        detect_th=0.005
        mlp_dim=16
        mlp_name="7scenes_stairs"    # 7scenes_stairs, 7scenes_stairs_pgt
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
        ace_ckpt="/home/koki/code/cc/feature_3dgs_2/data/ace_models/7Scenes_pgt/$scene_name.pt"
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
        save_match=0
        save_render_img=0
}
{
    out_name="1_${iterations}_${score_loss}_${score_scale}_${num_kpts}_${detect_th}_${mlp_dim}_${mlp_name}"
    ###############
    test_name="${test_iteration}_${loc_sp_th}_${loc_lg_th}_"
    test_name+="${score_hist}_${max_num_kpt}_${kernel_size}_${stop_kpt_num}_${pnp_option}"
    # test_name="test2"
}

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"

# ( bash z_scripts/tmp/train.sh )
if (( start_train )); then
    python train.py -s "$SOURSE_PATH" -i "rgb" -m "$OUT_PATH" --iterations $iterations \
                    --score_loss "$score_loss" --score_scale "$score_scale"  \
                    --num_kpts $num_kpts --detect_th $detect_th --mlp_dim $mlp_dim --mlp_name $mlp_name
fi


if (( start_render )); then
    detector_path="data/detector/7_scenes/pgt_7scenes_stairs/L2_0.0001_sptrain_normalizeRemoveNeg/epoch_138.pt"
    python render.py -m $OUT_PATH --iteration $iterations --view_num $render_num \
                        --sp_kpt $sp_kpt --sp_th $sp_th --detector_path $detector_path --hist $hist\
                        --match_precision_th $match_precision_th\
                        $([[ "$render_train" -eq 1 ]] && echo "--render_train") \
                        $([[ "$render_test" -eq 1 ]] && echo "--render_test") \
                        $([[ "$render_kpt_desc" -eq 1 ]] && echo "--render_kpt_desc") \
                        $([[ "$render_match" -eq 1 ]] && echo "--render_match")
fi

if (( start_loc )); then
    python -m z_localization.loc_inference_ace -m $OUT_PATH --test_name $test_name --iteration $test_iteration \
                                                --method $mlp_name --ace_ckpt $ace_ckpt \
                                                --match_type $match_type --sp_th $loc_sp_th --lg_th $loc_lg_th --kpt_hist $score_hist \
                                                --max_num_kpt $max_num_kpt --mlp_dim $mlp_dim \
                                                --kernel_size $kernel_size --stop_kpt_num $stop_kpt_num --pnp $pnp_option \
                                                $([[ "$save_match" -eq 1 ]] && echo "--save_match") \
                                                $([[ "$save_render_img" -eq 1 ]] && echo "--save_render_img")
fi