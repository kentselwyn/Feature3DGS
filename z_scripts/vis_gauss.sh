{
    data_name="7_scenes"
    scene_name="pgt_7scenes_stairs"
    
    iterations=10000
    # iterations=30000
    render_num=25
    out_name="twoPhase_all_weighted0.2_scale0.6"
    # out_name="pgt_7scenes_stairs_imrate:1_th:0.01_mlpdim:16_kptnum:1024_Scoreweighted_ScoreScale0.6_rgb_UseTrueRender"
}
# ( bash zenith_scripts/twoPhase.sh )
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"

# python gaussian_loader.py -m $OUT_PATH --iteration $iterations --skip_train --view_num $render_num
# render_desc, render_match
python vis_gauss.py -m $OUT_PATH --iteration $iterations --skip_train --view_num $render_num --render_desc
