# bash zenith_scripts/train.sh
{
    echo "start training..."
    SOURSE_PATH=$1
    feature_name="features/$2"
    out_name=$3
    img_name=$4
    score_loss=$5
    score_scale=$6
    mlp_dim=$7
}

export MLP_DIM=$mlp_dim
OUT_PATH="$SOURSE_PATH/outputs/$out_name"

python train.py -s "$SOURSE_PATH" -m "$OUT_PATH" -i "$img_name" -f "$feature_name" --iterations 8000 \
                --score_loss "$score_loss" --score_scale $score_scale
