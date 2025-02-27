##### training parameters######################################################################## 
{
    ###############################################
    # 7_scenes
    # Cambridge
    data_name="Cambridge"
    ###############################################
    # pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_pumpkin, pgt7scenes_redkitchen, pgt_7scenes_stairs
    # Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
    # scene_name="Cambridge_OldHospital" 
    scene_name=$1
    ###############################################
    # all
    # pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_pumpkin, pgt_7scenes_redkitchen, pgt_7scenes_stairs
    # Cambridge
    # Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
    # mlp_method="Cambridge_OldHospital"
    mlp_method=$1
    ###### dataset build #########################################
    resize_num=1
    th=0.01
    mlp_dim=16
    max_num_keypoints=1024
    # data_ImgName="imrate:${resize_num}_th:${th}_kptnum:${max_num_keypoints}"
    data_ImgName="None"
    # feature name
    # feat_option="small"
    feat_name="${mlp_method}_imrate:${resize_num}_th:${th}_"
    feat_name+="mlpdim:${mlp_dim}_kptnum:${max_num_keypoints}_rgb"
}
# ( bash zenith_scripts/DatasetBuild_Train_Loc2.sh )
{
    ###### gaussian training #########################################
    score_loss="weighted" # L2, weighted, L1
    score_scale=0.6
    feature_opa=0
    iterations=30000
    # train_ImgName="imrate:${resize_num}_th:${th}_kptnum:${max_num_keypoints}"
    train_ImgName="rgb"
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Warninggggggggggggggggggggggggggggggggggggggggg
    # traing_name_option="UseTrueRender_denseless6000_DenseInterval100"
    traing_name_option="UseTrueRender"
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Warninggggggggggggggggggggggggggggggggggggggggg
    # output name
    out_name="${mlp_method}_imrate:${resize_num}_th:${th}_"
    out_name+="mlpdim:${mlp_dim}_kptnum:${max_num_keypoints}_"
    out_name+="Score${score_loss}_ScoreScale${score_scale}_${train_ImgName}_${traing_name_option}"
    if (( feature_opa )); then
        out_name+="_FeatureOpa"
    fi
    # render
    render_num=20
    if_render=1
}


# ( bash zenith_scripts/DatasetBuild_Train_Loc2.sh )
########################################################################
build_data=1
start_train=1
start_loc=1
########################################################################


##### localization parameters ######################################################################## 
{
    rival=0
    save_match=0
    feat_from_img=0  # use score from gaussian, feature from image
    ###########################
    depth_render=0
    ###########################
    test_iteration=$iterations
    sp_th=0.01
    lg_th=0.01
    kpt_hist=0.95
    ransac_iters=20000
    kernel_size=15
    stop_kpt_num=50
    pnp_option="pycolmap" #iters, epnp, pycolmap

    # localization name
    test_name="iteration${test_iteration}_sp${sp_th}_lg${lg_th}_kptth0.01_"
    test_name+="kpthist${kpt_hist}_ransaciters${ransac_iters}_KptKernalSize${kernel_size}_"
    test_name+="stop_kpt_num${stop_kpt_num}_pnp_${pnp_option}_ace"

    if (( depth_render )); then
        test_name+="DepthRender"
    fi

    if (( feat_from_img )); then
        test_name+="FeatFromImg"
    fi

    if (( rival )); then
        test_name+="_rival_000"
    fi
}

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
feature_name="features/$feat_name"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"

######################################################################## dataset build ########################################################################
if (( build_data )); then
    s_path_train="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/${data_name}/${scene_name}/train"
    s_path_test="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/${data_name}/${scene_name}/test"
    python dataset_build.py --source_path "$s_path_train" --method "$mlp_method" --mlp_dim $mlp_dim  --images rgb \
                            --output_images "$data_ImgName" \
                            --resize_num $resize_num --th $th --max_num_keypoints $max_num_keypoints --name $feat_name
    python dataset_build.py --source_path "$s_path_test" --method "$mlp_method" --mlp_dim $mlp_dim --images rgb \
                            --output_images "$data_ImgName" \
                            --resize_num $resize_num --th $th --max_num_keypoints $max_num_keypoints --name $feat_name
fi

######################################################################## train ########################################################################
if (( start_train )); then
    export FEATURE_OPA=$feature_opa
    python train.py -s "$SOURSE_PATH" -i "$train_ImgName" -m "$OUT_PATH" -f "$feature_name" --iterations $iterations \
                    --score_loss "$score_loss" --score_scale "$score_scale"
    if (( if_render )); then
        python render.py -m $OUT_PATH --iteration $iterations --skip_train --view_num $render_num
    fi
fi

######################################################################## localization ########################################################################
if (( start_loc )); then
    export MLP_DIM=$mlp_dim
    export DEPTH_RENSER=$depth_render
    export FEATURE_OPA=$feature_opa
    if [[ "$data_name" == "7_scenes" ]]; then
        ace_ckpt="/home/koki/code/cc/feature_3dgs_2/data/ace_models/7Scenes_pgt/$scene_name.pt"
    elif [[ "$data_name" == "Cambridge" ]]; then
        ace_ckpt="/home/koki/code/cc/feature_3dgs_2/data/ace_models/Cambridge/$scene_name.pt"
    else
        echo "Dataset not exist"
    fi

    if (( save_match )); then
        python loc_inference_ace.py --save_match \
                                    -m $OUT_PATH --ace_ckpt $ace_ckpt --iteration $test_iteration \
                                    --mlp_dim $mlp_dim --mlp_method $mlp_method --lg_th $lg_th \
                                    --kernel_size $kernel_size --sp_th $sp_th --ransac_iters $ransac_iters \
                                    --stop_kpt_num $stop_kpt_num --pnp $pnp_option --kpt_hist $kpt_hist \
                                    --test_name "$test_name" --rival "$rival" --feat_from_img "$feat_from_img"
    else
        python loc_inference_ace.py -m $OUT_PATH --ace_ckpt $ace_ckpt --iteration $test_iteration \
                                    --mlp_dim $mlp_dim --mlp_method $mlp_method --lg_th $lg_th \
                                    --kernel_size $kernel_size --sp_th $sp_th --ransac_iters $ransac_iters \
                                    --stop_kpt_num $stop_kpt_num --pnp $pnp_option --kpt_hist $kpt_hist \
                                    --test_name "$test_name" --rival "$rival" --feat_from_img "$feat_from_img"
    fi
fi
