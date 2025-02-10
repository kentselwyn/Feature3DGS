
data_name=$1
scene_name=$2
mlp_method=$3
mlp_dim=$4

s_path_train="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/${data_name}/${scene_name}/train"
s_path_test="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/${data_name}/${scene_name}/test"
python dataset_build.py --source_path "$s_path_train" --method "$mlp_method" --mlp_dim $mlp_dim  --images rgb
python dataset_build.py --source_path "$s_path_test" --method "$mlp_method" --mlp_dim $mlp_dim --images rgb

# bash zenith_scripts/dataset_build.sh 7_scenes pgt_7scenes_stairs augment_pgt_7scenes_stairs 16
