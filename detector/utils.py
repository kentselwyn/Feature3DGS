import torch
import torch.nn.functional as F

def weighted_l2(fmap, gtmap, alpha=0.2):
    weights = torch.exp(alpha * gtmap)
    return torch.mean(weights * (fmap - gtmap) ** 2)

def sharpen_loss(fmap):
    # fmap: shape [B, 1, H, W] (單通道 heatmap)
    laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                       [1, -4, 1],
                                       [0, 1, 0]]]], dtype=fmap.dtype, device=fmap.device)
    edge = F.conv2d(fmap, laplacian_kernel, padding=1)
    return torch.mean(edge ** 2)

def combined_loss(fmap, gtmap, alpha=0.2, beta=0.01):
    return weighted_l2(fmap, gtmap, alpha) + beta * sharpen_loss(fmap)


def laplacian_sharpness_loss(pred):
    laplacian_kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        dtype=torch.float32, device=pred.device
    ).view(1, 1, 3, 3)
    lap = F.conv2d(pred, laplacian_kernel, padding=1)
    return lap.abs().mean()


def local_maxima_map(x, kernel_size=3):
    """
    x: Tensor of shape [B, 1, H, W]
    return: binary mask of local peaks [B, 1, H, W]
    """
    pad = kernel_size // 2
    max_pool = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    peak_mask = (x == max_pool).float()
    return peak_mask


def peakness_loss(pred):
    peak_mask = local_maxima_map(pred)
    return - (peak_mask * pred).mean()


def normalize_to_01(x, eps=1e-6):
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def remove_neg_and_normalize(gt, eps=1e-6):
    """
    1. clip all negative values to 0
    2. normalize each gt score map in batch to [0, 1]
    """
    gt = gt.clamp(min=0.0)  # step 1: remove negatives
    B = gt.shape[0]
    gt_min = gt.view(B, -1).amin(dim=1, keepdim=True).view(B, 1, 1, 1)
    gt_max = gt.view(B, -1).amax(dim=1, keepdim=True).view(B, 1, 1, 1)
    return (gt - gt_min) / (gt_max - gt_min + eps)
