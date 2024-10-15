# bash zenith_scripts/render2.sh
SOURSE_PATH1="/home/koki/code/cc/feature_3dgs_2/all_data/scene0713_00/A"
SOURSE_PATH2="/home/koki/code/cc/feature_3dgs_2/all_data/scene0724_00/A"

out_name1="9"
out_name2="8"

iteration=10000



python render.py -s "$SOURSE_PATH1" -m "$SOURSE_PATH1/outputs/$out_name1" --iteration "$iteration"
python render.py -s "$SOURSE_PATH1" -m "$SOURSE_PATH1/outputs/$out_name2" --iteration "$iteration"

# -f "$feature_name" -i "$img_name"
cp "$0" "$SOURSE_PATH1/outputs/$out_name1"
cp "$0" "$SOURSE_PATH1/outputs/$out_name2"




python render.py -s "$SOURSE_PATH2" -m "$SOURSE_PATH2/outputs/$out_name1" --iteration "$iteration"
python render.py -s "$SOURSE_PATH2" -m "$SOURSE_PATH2/outputs/$out_name2" --iteration "$iteration"

# -f "$feature_name" -i "$img_name"
cp "$0" "$SOURSE_PATH2/outputs/$out_name1"
cp "$0" "$SOURSE_PATH2/outputs/$out_name2"