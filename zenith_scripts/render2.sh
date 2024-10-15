# bash zenith_scripts/render2.sh
SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/all_data/scene0724_00/A"

out_name="5"

iteration=10000


OUT_PATH="$SOURSE_PATH/outputs/$out_name"


python render.py -s "$SOURSE_PATH" -m "$OUT_PATH" --iteration "$iteration"

# -f "$feature_name" -i "$img_name"
cp "$0" "$OUT_PATH"