# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms.functional as F

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False
import torchvision.transforms as tvf
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def img_to_arr( img ):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

def crop_center(image, crop_width, crop_height):
    img_width, img_height = image.size
    left = (img_width - crop_width) / 2
    top = (img_height - crop_height) / 2
    right = (img_width + crop_width) / 2
    bottom = (img_height + crop_height) / 2

    return image.crop((left, top, right, bottom))

def load_images_crop(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        img = crop_center(img, 1080, 1080)
        W1, H1 = img.size

        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], 
                         true_shape=np.int32([img.size[::-1]]), 
                         idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs




def convert_tensor_to_dust3r_format(tensor, size=512, square_ok=False, idx=0):
    """
    Convert a PyTorch tensor to the format expected by DUSt3R
    
    Parameters:
    - tensor: A PyTorch tensor with shape [1, 3, H, W] with values in range [0, 1]
    - size: Target size for resizing (default 512)
    - square_ok: Whether square outputs are acceptable
    - idx: The index to assign to this image
    
    Returns:
    - A dictionary in the format expected by DUSt3R
    """
    # Check input shape
    assert tensor.dim() == 4 and tensor.shape[0] == 1 and tensor.shape[1] == 3, \
        f"Expected tensor of shape [1, 3, H, W], got {tensor.shape}"
    
    # Get original height and width
    H, W = tensor.shape[2], tensor.shape[3]
    
    # Remove batch dimension for processing
    img_tensor = tensor.squeeze(0)
    
    # Implement resizing logic based on the original function
    if size == 224:
        # Resize short side to 224 (then crop)
        scale = size / min(H, W)
        new_size = [round(H * scale), round(W * scale)]
        img_tensor = F.resize(img_tensor, new_size, antialias=True)
    else:
        # Resize long side to specified size (default 512)
        scale = size / max(H, W)
        new_size = [round(H * scale), round(W * scale)]
        img_tensor = F.resize(img_tensor, new_size, antialias=True)
    
    # Update dimensions after resize
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    cx, cy = W // 2, H // 2
    
    # Implement cropping logic based on the original function
    if size == 224:
        half = min(cx, cy)
        img_tensor = F.crop(img_tensor, cy - half, cx - half, 2 * half, 2 * half)
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not square_ok and W == H:
            halfh = int(3 * halfw / 4)
        img_tensor = F.crop(img_tensor, cy - halfh, cx - halfw, 2 * halfh, 2 * halfw)
    
    # Get final dimensions
    _, H_final, W_final = img_tensor.shape
    
    # Apply normalization with mean and std of (0.5, 0.5, 0.5)
    normalized_tensor = F.normalize(img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Add batch dimension back
    normalized_tensor = normalized_tensor.unsqueeze(0)
    
    # Create the dictionary
    img_dict = {
        'img': normalized_tensor,
        'true_shape': np.int32([(H_final, W_final)]),
        'idx': idx,
        'instance': str(idx)
    }
    
    return img_dict


if __name__=="__main__":
    # Example usage
    your_tensor = torch.ones(1, 3, 480, 640) * 0.5  # Example tensor in range [0, 1]
    converted_dict = convert_tensor_to_dust3r_format(your_tensor)

    # If you need a list of images as returned by load_images
    images_list = [converted_dict]

