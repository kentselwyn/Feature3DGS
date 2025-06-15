# bash zenith_scripts/render.sh
SOURSE_PATH=$1
out_name=$2
img_name=$3
mlp_dim=$4

export MLP_DIM=$mlp_dim
iteration=8000

python render.py -s "$SOURSE_PATH" -i $img_name -m "$SOURSE_PATH/outputs/$out_name" \
                --iteration "$iteration" --skip_train --skip_test --pairs
