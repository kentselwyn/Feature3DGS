SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs"
# 7 scenes
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_chess"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_fire"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_heads"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_office"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_pumpkin"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_redkitchen"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs"
# Cambridge
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/Cambridge/Cambridge_KingsCollege"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/Cambridge/Cambridge_OldHospital"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/Cambridge/Cambridge_ShopFacade"
# "/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/Cambridge/Cambridge_StMarysChurch"
name="SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_rgb"
OUT_PATH="$SOURSE_PATH/outputs/$name"

python loc_inference.py -m $OUT_PATH



# bash zenith_scripts/loc_inference.sh

# 1. export MLP dataset ->                                  (python -m mlp.export --method sp)
# 2. train MLP(use b5,b6) ->                                (nohup python -u -m mlp.train --dim 16 > /home/koki/code/cc/feature_3dgs_2/log_dim16_fire.txt 2>&1 &)
# 3. use MLP build gaussian feature set(tarin+test) ->      (bash zenith_scripts/dataset_build.sh)
# 4. train gaussian ->                                      (bash zenith_scripts/train_7scenes.sh)
# 5. localization inference(need to change to depth median) (bash zenith_scripts/loc_inference.sh)

