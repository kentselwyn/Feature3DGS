import os
import cv2
import numpy as np


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



