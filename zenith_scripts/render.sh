# bash zenith_scripts/render.sh
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/all_data/scene0724_00/A"

out_name="11"
iteration=10000

python render.py -s "$SOURSE_PATH" -m "$SOURSE_PATH/outputs/$out_name" --iteration "$iteration"


cp "$0" "$SOURSE_PATH/outputs/$out_name"
