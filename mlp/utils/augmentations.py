import cv2
import torch
import random
import numpy as np
from typing import Union
import albumentations as A
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2


class IdentityTransform(A.ImageOnlyTransform):
    def apply(self, img, **params):
        return img

    def get_transform_init_args_names(self):
        return ()


class RandomAdditiveShade(A.ImageOnlyTransform):
    def __init__(
        self,
        nb_ellipses=10,
        transparency_limit=[-0.5, 0.8],
        kernel_size_limit=[150, 350],
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.nb_ellipses = nb_ellipses
        self.transparency_limit = transparency_limit
        self.kernel_size_limit = kernel_size_limit

    def apply(self, img, **params):
        if img.dtype == np.float32:
            shaded = self._py_additive_shade(img * 255.0)
            shaded /= 255.0
        elif img.dtype == np.uint8:
            shaded = self._py_additive_shade(img.astype(np.float32))
            shaded = shaded.astype(np.uint8)
        else:
            raise NotImplementedError(
                f"Data augmentation not available for type: {img.dtype}"
            )
        return shaded

    def _py_additive_shade(self, img):
        grayscale = len(img.shape) == 2
        if grayscale:
            img = img[None]
        min_dim = min(img.shape[:2]) / 4
        mask = np.zeros(img.shape[:2], img.dtype)
        for i in range(self.nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, img.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = np.random.uniform(*self.transparency_limit)
        ks = np.random.randint(*self.kernel_size_limit)
        if (ks % 2) == 0:  # kernel_size has to be odd
            ks += 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (ks, ks), 0)
        shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.0)
        out = np.clip(shaded, 0, 255)
        if grayscale:
            out = out.squeeze(0)
        return out

    def get_transform_init_args_names(self):
        return "transparency_limit", "kernel_size_limit", "nb_ellipses"

def kw(entry: Union[float, dict], n=None, **default):
    if not isinstance(entry, dict):
        entry = {"p": entry}
    entry = OmegaConf.create(entry)
    if n is not None:
        entry = default.get(n, entry)
    return OmegaConf.merge(default, entry)

def kwi(entry: Union[float, dict], n=None, **default):
    conf = kw(entry, n=n, **default)
    return {k: conf[k] for k in set(default.keys()).union(set(["p"]))}

def replay_str(transforms, s="Replay:\n", log_inactive=True):
    for t in transforms:
        if "transforms" in t.keys():
            s = replay_str(t["transforms"], s=s)
        elif t["applied"] or log_inactive:
            s += t["__class_fullname__"] + " " + str(t["applied"]) + "\n"
    return s


class BaseAugmentation(object):
    base_default_conf = {
        "name": "???",
        "shuffle": False,
        "p": 1.0,
        "verbose": False,
        "dtype": "uint8",  # (byte, float)
    }

    default_conf = {}

    def __init__(self, conf={}):
        """Perform some logic and call the _init method of the child model."""
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf),
        )
        OmegaConf.set_struct(default_conf, True)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(self.conf, True)
        self._init(self.conf)        
        self.conf = OmegaConf.merge(self.conf, conf)
        if self.conf.verbose:
            self.compose = A.ReplayCompose
        else:
            self.compose = A.Compose
        
        if self.conf.dtype == "uint8":
            self.dtype = np.uint8
            self.preprocess = A.FromFloat(always_apply=True, dtype="uint8")
            self.postprocess = A.ToFloat(always_apply=True)
        elif self.conf.dtype == "float32":
            self.dtype = np.float32
            self.preprocess = A.ToFloat(always_apply=True)
            self.postprocess = IdentityTransform()
        else:
            raise ValueError(f"Unsupported dtype {self.conf.dtype}")
        
        self.to_tensor = ToTensorV2()

    def _init(self, conf):
        """Child class overwrites this, setting up a list of transforms"""
        self.transforms = []

    def __call__(self, image, return_tensor=False):
        """image as HW or HWC"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        data = {"image": image}
        if image.dtype != self.dtype:
            data = self.preprocess(**data)
        transforms = self.transforms
        if self.conf.shuffle:
            order = [i for i, _ in enumerate(transforms)]
            np.random.shuffle(order)
            transforms = [transforms[i] for i in order]
        transformed = self.compose(transforms, p=self.conf.p)(**data)
        if self.conf.verbose:
            print(replay_str(transformed["replay"]["transforms"]))
        transformed = self.postprocess(**transformed)
        if return_tensor:
            return self.to_tensor(**transformed)["image"]
        else:
            return transformed["image"]


class IdentityAugmentation(BaseAugmentation):
    default_conf = {}
    def _init(self, conf):
        self.transforms = [IdentityTransform(p=1.0)]


class DarkAugmentation(BaseAugmentation):
    default_conf = {"p": 0.75}
    def _init(self, conf):
        bright_contr = 0.5
        blur = 0.1
        random_gamma = 0.1
        hue = 0.1
        self.transforms = [
            A.RandomRain(p=0.2),
            A.RandomBrightnessContrast(
                **kw(
                    bright_contr,
                    brightness_limit=(-0.4, 0.0),
                    contrast_limit=(-0.3, 0.0),
                )
            ),
            A.OneOf(
                [
                    A.Blur(**kwi(blur, p=0.1, blur_limit=(3, 9), n="blur")),
                    A.MotionBlur(
                        **kwi(
                            blur,
                            p=0.2,
                            blur_limit=(3, 25),
                            allow_shifted=False,
                            n="motion_blur",
                        )
                    ),
                    A.ISONoise(),
                    A.ImageCompression(),
                ],
                **kwi(blur, p=0.1),
            ),
            A.RandomGamma(**kw(random_gamma, gamma_limit=(15, 65))),
            A.OneOf(
                [
                    A.Equalize(),
                    A.CLAHE(p=0.2),
                    A.ToGray(),
                    A.ToSepia(p=0.1),
                    A.HueSaturationValue(**kw(hue, val_shift_limit=(-100, -40))),
                ],
                p=0.5,
            ),
        ]


class LGAugmentation(BaseAugmentation):
    default_conf = {"p": 0.95}
    def _init(self, conf):
        self.transforms = [
            A.RandomGamma(p=0.3, gamma_limit=(15, 65)),
            A.HueSaturationValue(p=0.3, val_shift_limit=(-100, -40)),
            A.OneOf(
                [
                    A.Blur(blur_limit=(3, 9)),
                    A.MotionBlur(blur_limit=(3, 25), allow_shifted=False),
                    A.ISONoise(),
                    A.ImageCompression(),
                ],
                p=0.3,
            ),
            A.Blur(p=0.3, blur_limit=(3, 9)),
            A.MotionBlur(p=0.3, blur_limit=(3, 25), allow_shifted=False),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=(-0.4, 0.0), contrast_limit=(-0.3, 0.0)),
            A.CLAHE(p=0.2),
        ]


class myAugmentation(BaseAugmentation):
    default_conf = {"p": 0.95}
    def _init(self, conf):
        self.transforms = [
            A.RandomGamma(p=0.1, gamma_limit=(45, 75)),
            A.HueSaturationValue(p=0.1, val_shift_limit=(-40, -20)),
            A.OneOf(
                [
                    A.Blur(blur_limit=(3, 7)),
                    A.ISONoise(),
                    # A.ImageCompression(),
                ],
                p=0.1,
            ),
            A.Blur(p=0.1, blur_limit=(3, 7)),
            A.MotionBlur(p=0.1, blur_limit=(3, 25), allow_shifted=False),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.15, 0.0), contrast_limit=(-0.15, 0.0)),
            A.CLAHE(p=0.2),
        ]


class testAugmentation(BaseAugmentation):
    default_conf = {"p": 1.}
    def _init(self, conf):
        self.transforms = [
            # A.MotionBlur(blur_limit=(3, 25), allow_shifted=False, p=1.),
            # A.Blur(blur_limit=(3, 11), p=1.),
            # A.RandomBrightnessContrast(
            #     p=1, brightness_limit=(-0.16, -0.15), contrast_limit=(-0.16, -0.15)),
            # A.HueSaturationValue(p=1, val_shift_limit=(-40, -35)),
            # A.RandomGamma(p=1, gamma_limit=(70, 75))
            A.CLAHE(p=1.),
        ]


# 1. Bright/Contrast/Gamma: Emphasize brightness-contrast changes
class aug0(BaseAugmentation):
    default_conf = {"p": 0.98}

    def _init(self, conf):
        # self.transforms = [
        #     # Lighting and shadow effects with higher probabilities
        #     A.RandomShadow(
        #         shadow_roi=(0, 0, 1, 1),
        #         num_shadows_lower=1,
        #         num_shadows_upper=2,
        #         shadow_dimension=5,
        #         p=0.9  # Increased from 0.7
        #     ),
        #     A.RandomToneCurve(scale=0.3, p=0.85),  # Increased from 0.6
        #     A.RandomGamma(
        #         gamma_limit=(80, 120),
        #         p=0.85  # Increased from 0.6
        #     ),
        # ]
        self.transforms = [
            # Replaced shadow with more feature-friendly transforms
            # A.ColorJitter(
            #     brightness=0.4,
            #     contrast=0.4,
            #     saturation=0.4,
            #     hue=0.2,
            #     p=1.0
            # ),
            A.FancyPCA(  # Adds photometric noise through PCA color augmentation
                alpha=0.4,
                p=1.0
            ),
            A.RandomToneCurve(
                scale=0.3, 
                p=0.85
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=1.0
            ),
            # A.Equalize(  # Histogram equalization for better contrast
            #     mode='cv',
            #     by_channels=True,
            #     p=1.0
            # ),
            # A.CLAHE(  # Added Contrast Limited Adaptive Histogram Equalization
            #     clip_limit=4.0,
            #     tile_grid_size=(8, 8),
            #     p=0.7
            # ),
        ]


# 2. Hue/Sat + RGB Shift: Emphasize color hue shifts
class aug1(BaseAugmentation):
    default_conf = {"p": 0.98}

    def _init(self, conf):
        self.transforms = [
            # Color-based transforms with higher probabilities
            A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
            ),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.99  # Increased from 0.7
            ),
        ]

# 3. Posterize/Solarize + Equalize: Emphasize “creative” color effects
class aug2(BaseAugmentation):
    default_conf = {"p": 0.98}

    def _init(self, conf):
        self.transforms = [
            # Texture-enhancing transforms with higher probabilities
            A.Sharpen(
                    alpha=(0.4, 0.5),
                    lightness=(0.8, 1.0),
                    p=1.0
                ),
            A.Emboss(
                    alpha=(0.2, 0.5),
                    strength=(0.5, 1.0),
                    p=1.0
                ),
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=0.95  # Increased from 0.6
            ),
        ]

# 4. Channel Shuffle / Grayscale: Emphasize drastic color channel changes
class aug3(BaseAugmentation):
    default_conf = {"p": 0.98}

    def _init(self, conf):
        self.transforms = [
            # Weather-like effects with higher probabilities
            # A.RandomFog(
            #         fog_coef_lower=0.1,
            #         fog_coef_upper=0.2,
            #         alpha_coef=0.08,
            #         p=1.0
            # ),
            A.RandomRain(
                slant_lower=-2,
                slant_upper=2,
                drop_length=10,
                drop_width=1,
                drop_color=(200, 100, 100),
                p=1.0
            ),
            # A.RandomBrightnessContrast(
            #     brightness_limit=(-0.2, 0.2),
            #     contrast_limit=(-0.2, 0.2),
            #     p=0.98  # Increased from 0.6
            # ),
        ]


class aug4(BaseAugmentation):
    default_conf = {"p": 0.95}

    def _init(self, conf):
        self.transforms = [
            # Stronger noise and blur effects
            A.OneOf([
                A.GaussNoise(
                    var_limit=(100.0, 250.0),  # Increased from (10.0, 50.0)
                    mean=0,
                    per_channel=True,  # Added per-channel noise
                    p=1.0
                ),
                A.ISONoise(
                    color_shift=(0.05, 0.15),  # Increased from (0.01, 0.05)
                    intensity=(0.5, 1.0),      # Increased from (0.1, 0.5)
                    p=1.0
                ),
                A.MultiplicativeNoise(  # Added new noise type
                    multiplier=(0.7, 1.3),
                    per_channel=True,
                    elementwise=True,
                    p=1.0
                ),
            ], p=0.99),
            A.OneOf([
                A.GaussianBlur(
                    blur_limit=(7, 11),  # Increased from (3, 5)
                    p=1.0
                ),
                A.MotionBlur(  # Added motion blur
                    blur_limit=(15, 25),
                    p=1.0
                ),
                A.MedianBlur(  # Added median blur
                    blur_limit=7,
                    p=1.0
                ),
            ], p=0.9),
        ]


augmentations = {
    "dark": DarkAugmentation,
    "lg": LGAugmentation,
    "identity": IdentityAugmentation,
    "myaug":myAugmentation,
    "0": aug0(),
    "1": aug1(),
    "2": aug2(),
    "3": aug3(),
    "4": aug4(),
}


# python -m mlp.augmentations
if __name__=="__main__":
    random.seed(180)
    image = cv2.imread("/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs/train/rgb/seq-02-frame-000010.color.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imsave('/home/koki/code/cc/feature_3dgs_2/test_imgs/100.png', image)

    aug_0 = aug0()
    img = aug_0(image=image)
    plt.imsave('/home/koki/code/cc/feature_3dgs_2/test_imgs/0.png', img)

    aug_1 = aug1()
    img = aug_1(image=image)
    plt.imsave('/home/koki/code/cc/feature_3dgs_2/test_imgs/1.png', img)

    aug_2 = aug2()
    img = aug_2(image=image)
    plt.imsave('/home/koki/code/cc/feature_3dgs_2/test_imgs/2.png', img)

    aug_3 = aug3()
    img = aug_3(image=image)
    plt.imsave('/home/koki/code/cc/feature_3dgs_2/test_imgs/3.png', img)

    aug_4 = aug4()
    img = aug_4(image=image)
    plt.imsave('/home/koki/code/cc/feature_3dgs_2/test_imgs/4.png', img)
