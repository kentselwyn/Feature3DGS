##### training parameters######################################################################## 
{
    ###############################################
    # 7scenes
    # Cambridge
    data_name="7_scenes"
    ###############################################
    # pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_pumpkin, pgt_7scenes_redkitchen, pgt_7scenes_stairs
    # Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
    scene_name="scene_stairs" 
    ###############################################
    # all, dataset_7scenes
    # pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_pumpkin, pgt_7scenes_redkitchen, pgt_7scenes_stairs
    # Cambridge
    # Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
    mlp_method="pgt_7scenes_stairs"
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

{
    ###### gaussian training #########################################
    score_loss="L2" # L2, weighted, L1
    # score_loss="L2"
    score_scale=0.6
    feature_opa=0
    iterations=30000
    render_num=20
    # train_ImgName="imrate:${resize_num}_th:${th}_kptnum:${max_num_keypoints}"
    train_ImgName="images"
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
}

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/$data_name/$scene_name/train"
feature_name="features/$feat_name"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"



start_train=0
if_render=1
if (( start_train )); then
    export MLP_DIM=$mlp_dim
    export FEATURE_OPA=$feature_opa
    python train.py -s "$SOURSE_PATH" -i "$train_ImgName" -m "$OUT_PATH" -f "$feature_name" --iterations $iterations \
                    --score_loss "$score_loss" --score_scale "$score_scale"
fi
if (( if_render )); then
    python render.py -m $OUT_PATH --iteration $iterations --skip_train --view_num $render_num
fi

# ( bash zenith_scripts/train_cpr.sh )