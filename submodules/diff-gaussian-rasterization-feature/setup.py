#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# pip install submodules/diff-gaussian-rasterization-feature
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            #extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            ### To fix illegal memory issue
            extra_compile_args={"nvcc": [
                "-Xcompiler", 
                "-fno-gnu-unique",
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
    ],
    cmdclass={'build_ext': BuildExtension}
)



###
            # extra_compile_args={"nvcc": ["-O0", "-Xcompiler", "-fPIC", "-G", "-g", 
            #                              "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")],
            #     "cxx": ["-g"]  # Generate debug info for C++ code
            
            # extra_link_args=["-shared"]


# /home/koki/code/cc/feature_3dgs/submodules/diff-gaussian-rasterization-feature/setup.py
# /home/koki/code/cc/feature_3dgs/submodules/diff-gaussian-rasterization-feature/setup.py