name="imrate:2_th:0.01_mlpdim:16"
iteration=7000

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/all_data/scene0755_00/A"
OUT_PATH="$SOURSE_PATH/outputs/$name"

IMAGE_NAME=$name
FEATURE_NAME=$name

python render.py -s "$SOURSE_PATH" -m "$OUT_PATH" -f "$FEATURE_NAME" -i "$IMAGE_NAME" --iteration "$iteration" --novel_view --multi_interpolate

