data_name="7_scenes"
# scene_name=$1
scene_name="scene_stairs"
# augmentation=
img_save=0
num_workers=4
multiple=1

if (( img_save )); then
    python -m mlp.export --method SP --data_name $data_name --scene_name $scene_name \
                        --augment $augmentation --num_workers $num_workers --multiple $multiple --img_save
else
    python -m mlp.export --method SP --data_name $data_name --scene_name $scene_name \
                        --augment $augmentation --num_workers $num_workers --multiple $multiple
fi                        
# bash zenith_scripts/export.sh 7_scenes pgt_7scenes_stairs lg
