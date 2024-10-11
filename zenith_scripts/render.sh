

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/all_data/scene0755_00/A"
OUT_PATH="/home/koki/code/cc/feature_3dgs_2/all_data/scene0755_00/A/outputs/imrate:2_th:0.01_mlpdim:16"
FEATURE_NAME="imrate:2_th:0.01_mlpdim:16"
IMAGE_NAME="imrate:2_th:0.01_mlpdim:16"
iteration=7000

python render.py -s "$SOURSE_PATH" -m "$OUT_PATH" --images "$IMAGE_NAME" -f "$FEATURE_NAME" --iteration $iteration --novel_view --multi_interpolate




