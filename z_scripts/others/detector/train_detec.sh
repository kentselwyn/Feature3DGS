{
    ###############################################
    # 7_scenes
    # Cambridge
    data_name="7_scenes"
    ###############################################
    # pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_pumpkin, pgt_7scenes_redkitchen, pgt_7scenes_stairs
    # Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
    ss="$1"
    scene_name="scene_${ss}"
    ###############################################
    # all, dataset_7scenes
    # pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_pumpkin, pgt_7scenes_redkitchen, pgt_7scenes_stairs
    # Cambridge
    # Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
    mlp_method="pgt_7scenes_${ss}"
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
}


SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/GS-CPR/$data_name/$scene_name/train"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"



python train_detector.py -m "$OUT_PATH" --iterations $iterations 



# bash /home/koki/code/cc/feature_3dgs_2/zenith_scripts/train_detec.sh

