# Define all 7 Scenes datasets
datasets=("pgt_7scenes_fire" "pgt_7scenes_heads" "pgt_7scenes_office" "pgt_7scenes_pumpkin" "pgt_7scenes_redkitchen" "pgt_7scenes_stairs")

# Set profiling flag (set to true to enable profiling, false to disable)
ENABLE_PROFILING=false
# Profiling mode: "lightweight" (faster, sample batches) or "full" (slower, complete epochs)
PROFILE_MODE="lightweight"
# GPU preloading: true for faster training (more GPU memory), false for lower GPU memory usage
GPU_PRELOAD=true

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    echo "Training on dataset: $dataset"
    path="/work/u8351896/7scenes/$dataset/"
    out_path="/work/u8351896/7scenes/$dataset/mlp/"
    
    # Check if dataset directory exists
    if [ -d "$path" ]; then
        echo "Processing $dataset..."
        
        # Build command with GPU preloading option
        if [ "$GPU_PRELOAD" = true ]; then
            GPU_FLAG="--preload_gpu"
        else
            GPU_FLAG="--no_preload_gpu"
        fi
        
        if [ "$ENABLE_PROFILING" = true ]; then
            echo "Running with $PROFILE_MODE profiling enabled (GPU preload: $GPU_PRELOAD)..."
            python -u -m mlp.train --path $path --out_path $out_path --epochs 20 --profile --profile_mode $PROFILE_MODE $GPU_FLAG
        else
            echo "Running without profiling (GPU preload: $GPU_PRELOAD)..."
            python -u -m mlp.train --path $path --out_path $out_path --batch_size 512 $GPU_FLAG
        fi
        echo "Completed training for $dataset"
        echo "----------------------------------------"
    else
        echo "Warning: Dataset directory $path does not exist, skipping..."
    fi
done

echo "All datasets processed!"

# ( bash z_scripts/mlp/train.sh )
