import os
import cv2
import numpy as np
from PIL import Image
from scene.gaussian_model import GaussianModel

def get_low_resolution(name, folder_path):
    path = f"{folder_path}/{name}/images"
    out_path = f"{folder_path}/{name}/images_low_resolution"
    os.makedirs(out_path, exist_ok=True)
    images = os.listdir(path)
    for img in images:
        img_path = f"{path}/{img}"
        image = cv2.imread(img_path)
        scaling_factor = 0.25
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        downscaled = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        blurred = cv2.resize(downscaled, original_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f"{out_path}/{img}", blurred)


def npz():
    intrinsic_path = "/home/koki/code/cc/feature_3dgs_2/img_match/scannet_test_1500_info/test.npz"
    x = dict(np.load(intrinsic_path))
    z=x['name']

    for i in range(20):
        print(f'========={z[15 *i, 0]}===========')
        print(z[15*i: 15*(i+1), 2:])


def jpg2png():
    # Specify the folder path here
    folder_path = 'data/flower1/input'

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter out all JPG files
    jpg_files = [file for file in files if file.lower().endswith('.jpg')]

    # Convert each JPG file to PNG and delete the JPG file
    for jpg_file in jpg_files:
        # Create the full file path by joining folder path and file name
        jpg_file_path = os.path.join(folder_path, jpg_file)
        
        # Open the JPG file
        with Image.open(jpg_file_path) as im:
            # Create a new file name with .png extension
            png_file = os.path.splitext(jpg_file)[0] + '.png'
            
            # Create the full file path for the new PNG file
            png_file_path = os.path.join(folder_path, png_file)
            
            # Save the image in PNG format
            im.save(png_file_path)
        
        # Delete the original JPG file
        os.remove(jpg_file_path)
        print(f"{jpg_file} has been converted to {png_file} and the original JPG file has been deleted.")
        #print(f"{jpg_file} has been converted to {png_file}.")

    print('Finished!!!')



def vis_gaussian(path = "/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_00/A/outputs/outsp/point_cloud/iteration_7000/point_cloud.ply"):
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(path)

    print(gaussians.get_opacity.shape)

    breakpoint()

def move_image():
    input = "/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_01/A/all_images/images"
    files = os.listdir(input)
    files = sorted(files, key=lambda x: int(x.split('.')[0]))


    out = "/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_01/A/outputs/imrate:2_th:0.01_mlpdim:16/rendering/trains/ours_7000/image_orig"


    os.makedirs(f"{out}", exist_ok=True)

    for i, name in enumerate(files):
        path = f"{input}/{name}"
        img = Image.open(path)
        w,h = img.size
        img = img.resize((int(w/2), int(h/2)))
        img.save(f"{out}/{i:05d}.png")


