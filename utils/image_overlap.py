from imagededup.methods import CNN
import os

# Step 1: Initialize the CNN-based deduplicator
cnn = CNN()

# Step 2: Specify the folder containing images
image_dir = '/home/koki/code/cc/feature_3dgs_2/all_data/scene0708_00/A/outputs/2/rendering/test/ours_10000/image_gt'

# Step 3: Find duplicates (or similar images)
dp = cnn.find_duplicates(image_dir=image_dir, min_similarity_threshold=0.75)

keys = dp.keys()


for image, matches in dp.items():
    print(f"image:{image}: {matches}")


breakpoint()


# python image_overlap.py

