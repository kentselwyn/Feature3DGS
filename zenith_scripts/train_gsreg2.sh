# bash zenith_scripts/train_gsreg2.sh

directory="/home/koki/code/cc/feature_3dgs_2/gsreg/test/"
files=("$directory"*)
start_index=50
num_files=30
end_index=$((start_index + num_files - 1))


for ((i=start_index; i<=end_index && i<${#files[@]}; i++))
do
    scene_path="${files[$i]}"

    for item in A B
    do
        SOURSE_PATH="$scene_path/$item"

        score_loss=L2
        img_name="images"

        feature_name="SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6"
        feature_path="features/$feature_name"
        OUT_PATH="$SOURSE_PATH/outputs/$feature_name"
        ply_path="$OUT_PATH/point_cloud/iteration_8000/point_cloud.ply"

        if [ ! -e "$ply_path" ]; then
            echo "The $ply_path does not exist. start training."
            python dataset_build.py --input $SOURSE_PATH --feature_name $feature_name --resize_num 1 --max_num_keypoints 1024
            python train.py -s "$SOURSE_PATH" -m "$OUT_PATH" -i "$img_name" -f "$feature_path" --iterations 8000 --score_loss "$score_loss" --score_scale 0.6
            cp "$0" "$OUT_PATH"
        else
            echo "The $ply_path exists."
        fi
    done
done