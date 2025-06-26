start_train=1
start_render=0
start_loc=0
{
    data_name="7_scenes"
    scene_name="pgt_7scenes_stairs"
    iterations=30000
    score_loss="weighted" # L2, weighted, L1   
    score_scale=0.6
    out_name="test"
    render_num=100
}
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"

# ( bash z_scripts/train.sh )
if (( start_train )); then
    python train.py -s "$SOURSE_PATH" -i "rgb" -m "$OUT_PATH" --iterations $iterations \
                    --score_loss "$score_loss" --score_scale "$score_scale"
fi

if (( start_render )); then
    python render.py -m $OUT_PATH --iteration $iterations --skip_train --view_num $render_num
fi
