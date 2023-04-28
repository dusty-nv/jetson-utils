#!/usr/bin/env python3
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import argparse
import sys

try:
    import torch
except ImportError:
    print("failed to import torch - if you wish to test PyTorch interoperability, please install it")
    sys.exit(0)
        
from jetson_utils import cudaImage, cudaNormalize


# parse the command line
parser = argparse.ArgumentParser('Map cudaImage to PyTorch GPU tensor')

parser.add_argument("--width", type=int, default=6, help="width of the array (in pixels)")
parser.add_argument("--height", type=int, default=4, help="height of the array (in pixels)")
parser.add_argument("--channels", type=int, default=3, help="number of color channels (1, 3, or 4)")

args = parser.parse_args()
print(args)


def tensor_image_format(tensor):
    """
    Determine the cudaImage format string (eg 'rgb32f', 'rgba32f', ect) from a PyTorch tensor.
    Only float and uint8 tensors are supported because those datatypes are supported by cudaImage.
    """
    if tensor.dtype != torch.float32 and tensor.dtype != torch.uint8:
        raise ValueError(f"PyTorch tensor datatype should be torch.float32 or torch.uint8 (was {tensor.dtype})")
        
    if len(tensor.shape)>= 4:     # NCHW layout
        channels = tensor.shape[1]
    elif len(tensor.shape) == 3:   # CHW layout
        channels = tensor.shape[0]
    elif len(tensor.shape) == 2:   # HW layout
        channels = 1
    else:
        raise ValueError(f"PyTorch tensor should have at least 2 image dimensions (has {tensor.shape.length})")
        
    if channels == 1:   return 'gray32f' if tensor.dtype == torch.float32 else 'gray8'
    elif channels == 3: return 'rgb32f'  if tensor.dtype == torch.float32 else 'rgb8'
    elif channels == 4: return 'rgba32f' if tensor.dtype == torch.float32 else 'rgba8'
    
    raise ValueError(f"PyTorch tensor should have 1, 3, or 4 image channels (has {channels})")
    
    
# allocate a GPU tensor with NCHW layout (strided colors)
tensor = torch.rand(1, args.channels, args.height, args.width, dtype=torch.float32, device='cuda')

# transpose the channels to NHWC layout (interleaved colors)
tensor = tensor.to(memory_format=torch.channels_last)   # or tensor.permute(0, 3, 2, 1)

print("\nPyTorch tensor:")
print(type(tensor))
print(f"    -- ptr:   {hex(tensor.data_ptr())}")
print(f"    -- type:  {tensor.dtype}")
print(f"    -- shape: {tensor.shape}\n")
print(tensor)

# map to cudaImage using the same underlying memory (any changes will be reflected in the PyTorch tensor)
cuda_img = cudaImage(ptr=tensor.data_ptr(), width=tensor.shape[-1], height=tensor.shape[-2], format=tensor_image_format(tensor))

print("\ncudaImage:")
print(cuda_img)

# perform an operation on the cudaImage (scale it by 100x)
cudaNormalize(cuda_img, (0,1), cuda_img, (0,100))

# print out the PyTorch tensor again to show the values have been updated
print("\nPyTorch tensor (modified):\n")
print(tensor)


