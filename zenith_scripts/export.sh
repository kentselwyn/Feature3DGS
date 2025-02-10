
data_name=$1
scene_name=$2
augmentation="sequence"
num_workers=0
multiple=1

python -m mlp.export --method SP --data_name $data_name --scene_name $scene_name \
                    --augment $augmentation --num_workers $num_workers --multiple $multiple --img_save
# bash zenith_scripts/export.sh 7_scenes pgt_7scenes_stairs
