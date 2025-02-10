# bash zenith_scripts/render.sh
# SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test/scene0757_00/sfm_sample"
SOURSE_PATH=$1

img_name=$3
out_name="$2"
iteration=8000

python render.py -s "$SOURSE_PATH" -i $img_name -m "$SOURSE_PATH/outputs/$out_name" --iteration "$iteration" --skip_train --skip_test --pairs


cp "$0" "$SOURSE_PATH/outputs/$out_name"
