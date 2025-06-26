import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def load_image(path: Path, grayscale=False) -> torch.Tensor:
    image = read_image(path, grayscale=grayscale)
    return numpy_image_to_torch(image)


def load_image2(path: str, resize=None) -> torch.Tensor:
    image = Image.open(path)
    size = image.size
    if resize is not None:
        if type(resize) == int:
            image = image.resize([int(size[0]/resize), int(size[1]/resize)])
        elif type(resize)==list:
            image = image.resize(resize)
    transfrom = transforms.ToTensor()
    tensor = transfrom(image)
    return tensor
