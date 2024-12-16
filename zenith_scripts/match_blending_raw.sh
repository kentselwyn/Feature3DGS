# bash zenith_scripts/match_blending_raw.sh

type="truck"
out_name="raw_images_low_resolution"

OUT_PATH="/home/koki/code/cc/feature_3dgs_2/img_match/Else/tandt_db/$type/outputs/$out_name"

python match_blending_truck_raw.py -m $OUT_PATH
