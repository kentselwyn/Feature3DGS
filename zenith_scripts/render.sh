###############################################
# 7_scenes
# Cambridge
data_name="Cambridge"
###############################################
# pgt_7scenes_pumpkin, pgt_7scenes_stairs, pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_redkitchen
# Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
scene_name="Cambridge_KingsCollege"
###############################################
# all
# pgt_7scenes_pumpkin, augment_pgt_7scenes_stairs, pgt_7scenes_chess, pgt_7scenes_fire, pgt_7scenes_heads, pgt_7scenes_office, pgt_7scenes_redkitche
# Cambridge
# Cambridge_KingsCollege, Cambridge_OldHospital, Cambridge_ShopFacade, Cambridge_StMarysChurch
mlp_method="Cambridge"
###############################################
mlp_dim=16
view_num=20

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/$data_name/$scene_name"
out_name="${mlp_method}_imrate:1_th:0.01_mlpdim:${mlp_dim}_kptnum:1024_score0.6_rgb"
OUT_PATH="$SOURSE_PATH/outputs/$out_name"
python render.py -m $OUT_PATH --iteration 30000 --skip_train --view_num $view_num

# bash zenith_scripts/render.sh Cambridge Cambridge_KingsCollege Cambridge 32
# bash zenith_scripts/render.sh
