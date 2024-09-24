# test_cuda_debug.py
import torch
from diff_gaussian_rasterization import _C

def run_rasterization():
    input_tensor = torch.randn(1, 3, 64, 64).cuda()  # Example input tensor
    output = _C.rasterize_gaussians(input_tensor)
    print(output)


# python test_cuda_debug.py
if __name__ == "__main__":
    run_rasterization()
