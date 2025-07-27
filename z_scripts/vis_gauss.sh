{
    data_name="7scenes"
    
    # iterations=10000
    iterations=30000
    render_num=25
    out_name="no_rgb"
    # out_name="pgt_7scenes_stairs_imrate:1_th:0.01_mlpdim:16_kptnum:1024_Scoreweighted_ScoreScale0.6_rgb_UseTrueRender"
}

# Define all seven scenes in 7 Scenes dataset
scenes=("pgt_7scenes_chess" "pgt_7scenes_fire" "pgt_7scenes_heads" "pgt_7scenes_office" "pgt_7scenes_pumpkin" "pgt_7scenes_redkitchen" "pgt_7scenes_stairs")

# Loop through all scenes
for scene_name in "${scenes[@]}"; do
    echo "Processing scene: $scene_name"
    
    SOURSE_PATH="/work/u8351896/$data_name/$scene_name"
    OUT_PATH="$SOURSE_PATH/outputs/$out_name"
    
    # Check if the output path exists
    if [ ! -d "$OUT_PATH" ]; then
        echo "Warning: Output path does not exist for $scene_name: $OUT_PATH"
        echo "Skipping $scene_name..."
        continue
    fi
    
    echo "Running visualization for $scene_name..."
    # python gaussian_loader.py -m $OUT_PATH --iteration $iterations --skip_train --view_num $render_num
    # render_desc, render_match
    python vis_gauss.py -m $OUT_PATH --iteration $iterations --skip_train --view_num $render_num --render_desc
    
    echo "Completed visualization for $scene_name"
    echo "----------------------------------------"
done

echo "All scenes processed!"
