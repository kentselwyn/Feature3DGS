name="imrate:2_th:0.01_mlpdim:16"

SOURSE_PATH="/home/koki/code/cc/feature_3dgs_2/all_data/scene0755_00/A"
OUT_PATH="$SOURSE_PATH/outputs/$name"

IMAGE_NAME=$name
FEATURE_NAME=$name

python train.py -s "$SOURSE_PATH" -m "$OUT_PATH" -i "$IMAGE_NAME" -f "$FEATURE_NAME" --iterations 10000 --eval

