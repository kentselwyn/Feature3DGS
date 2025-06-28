start_train=0
start_render=1
start_loc=0
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
################### rendering ###################
    render_num=25
    sp_kpt=1024
    sp_th=0.0
#################################################
}
out_name="1_${iterations}_${score_loss}_${score_scale}_${num_kpts}_${detect_th}_${mlp_dim}_${mlp_name}"
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"

# ( bash z_scripts/train.sh )
if (( start_train )); then
    python train.py -s "$SOURSE_PATH" -i "rgb" -m "$OUT_PATH" --iterations $iterations \
                    --score_loss "$score_loss" --score_scale "$score_scale"  \
                    --num_kpts $num_kpts --detect_th $detect_th --mlp_dim $mlp_dim --mlp_name $mlp_name
fi

if (( start_render )); then
    python render.py -m $OUT_PATH --iteration $iterations --render_test --render_match --view_num $render_num \
                        --sp_kpt $sp_kpt --sp_th $sp_th
fi
