import torch
import torch.nn as nn

def simple_nms(scores, nms_radius : int):
    
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = (scores == max_pool(max_mask.float()) > 0)
    for _ in range(2):
        supplement_mask = max_pool(max_mask.float()) > 0
        supplement_scores = torch.where(supplement_mask, zeros, scores)
        new_max_mask = (supplement_scores == max_pool(supplement_scores))
        maximum_mask = max_mask | (new_max_mask & (~supplement_mask))

    result = torch.where(maximum_mask, scores, zeros)
    return result