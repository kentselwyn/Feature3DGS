
start_train_score=0
start_render_score=0
start_train_feature=1
start_loc=0
{
    data_name="7_scenes"
    scene_name="pgt_7scenes_stairs"
    
    iterations=10000
    score_loss="weighted" # L2, weighted, L1
    render_num=100
    score_scale=0.6
    out_name="twoPhase_all_weighted0.2_scale${score_scale}"
}
# ( bash zenith_scripts/twoPhasePipeline.sh )
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"

if (( start_train_score )); then
    python train_score.py -s "$SOURSE_PATH" -i "rgb" -m "$OUT_PATH" --iterations $iterations \
                    --score_loss "$score_loss" --score_scale "$score_scale"
fi

if (( start_render_score )); then
    python render_score.py -m $OUT_PATH --iteration $iterations --skip_train --view_num $render_num
    python render_score.py -m $OUT_PATH --iteration 5000        --skip_train --view_num $render_num
fi

if (( start_train_feature )); then
    python train_feature.py -s "$SOURSE_PATH" -i "rgb" -m "$OUT_PATH" --iterations $iterations \
                    --score_loss "$score_loss" --score_scale "$score_scale"
fi
