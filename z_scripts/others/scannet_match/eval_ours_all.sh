# bash zenith_scripts/scannet_match/eval_ours_all.sh
{   
    echo "Running evaluation..."
    # "SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_ScoreL2_ScoreScale0.6"
    # "SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_Scoreweighted_ScoreScale0.6"
    # "SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6"
    # "SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_Scoreweighted_ScoreScale0.6"
    # "SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_ScoreL2_ScoreScale0.6"
    # "SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_Scoreweighted_ScoreScale0.6"
    # "SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6"
    # "SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_Scoreweighted_ScoreScale0.6"
    # MatchResult_KptKernalSize15_KptHist0.9_LGth0.01
    # python -m z_scannet1500.eval_ours_all
}
old_exps=(
    "SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_ScoreL2_ScoreScale0.6"
    "SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6"
    "SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_ScoreL2_ScoreScale0.6"
)
new_exps=(
    "SP_imrate:1_th:0.01_mlpdim:8_kptnum:1024_Scoreweighted_ScoreScale0.6"
    "SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_Scoreweighted_ScoreScale0.6"
    "SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_ScoreL2_ScoreScale0.6"
    "SP_imrate:2_th:0.01_mlpdim:8_kptnum:1024_Scoreweighted_ScoreScale0.6"
    "SP_imrate:2_th:0.01_mlpdim:16_kptnum:1024_Scoreweighted_ScoreScale0.6"
)
all_path="/home/koki/code/cc/feature_3dgs_2/data/img_match/scannet_test"
save_path="/home/koki/code/cc/feature_3dgs_2/z_result"
match_name="MatchResult_KptKernalSize15_KptHist0.9_LGth0.01"
eval_all=1
eval_pcd_size=0

for out_name in "${old_exps[@]}"; do
    python -m z_scannet1500.eval_ours_all --all_path $all_path --save_path $save_path\
                                            --out_name $out_name --match_name $match_name\
                                            --eval_all $eval_all --eval_pcd_size $eval_pcd_size
done
