
data_name="Cambridge"
method="Cambridge"
img_name="rgb"

list=("Cambridge_KingsCollege" "Cambridge_OldHospital" "Cambridge_ShopFacade" "Cambridge_StMarysChurch")

for scene_name in "${list[@]}"; do
    SOURSE_PATH0="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/${data_name}/${scene_name}/train"
    python dataset_build.py --input "$SOURSE_PATH0" --mlp_dim 16 --th 0.01 --resize_num 1 --max_num_keypoints 1024 --method "$method" --images "$img_name"
    python dataset_build.py --input "$SOURSE_PATH0" --mlp_dim 32 --th 0.01 --resize_num 1 --max_num_keypoints 1024 --method "$method" --images "$img_name"
    python dataset_build.py --input "$SOURSE_PATH0" --mlp_dim 64 --th 0.01 --resize_num 1 --max_num_keypoints 1024 --method "$method" --images "$img_name"

    SOURSE_PATH1="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/${data_name}/${scene_name}/test"
    python dataset_build.py --input "$SOURSE_PATH1" --mlp_dim 16 --th 0.01 --resize_num 1 --max_num_keypoints 1024 --method "$method" --images "$img_name"
    python dataset_build.py --input "$SOURSE_PATH1" --mlp_dim 32 --th 0.01 --resize_num 1 --max_num_keypoints 1024 --method "$method" --images "$img_name"
    python dataset_build.py --input "$SOURSE_PATH1" --mlp_dim 64 --th 0.01 --resize_num 1 --max_num_keypoints 1024 --method "$method" --images "$img_name"
done

# bash zenith_scripts/dataset_build_Cambridge.sh
