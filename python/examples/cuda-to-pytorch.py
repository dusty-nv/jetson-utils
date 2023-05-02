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
    
from jetson_utils import cudaImage


# parse the command line
parser = argparse.ArgumentParser('Map cudaImage to PyTorch GPU tensor')

parser.add_argument("--width", type=int, default=4, help="width of the array (in pixels)")
parser.add_argument("--height", type=int, default=2, help="height of the array (in pixels)")
parser.add_argument("--format", type=str, default="rgb32f", help="format of the array (default rgb32f)")

args = parser.parse_args()
print(args)

# allocate cuda memory
cuda_img = cudaImage(width=args.width, height=args.height, format=args.format)

print(cuda_img)

# map to torch tensor using __cuda_array_interface__
tensor = torch.as_tensor(cuda_img, device='cuda')

print("\nPyTorch tensor:\n")
print(type(tensor))
print(f"    -- ptr:   {hex(tensor.data_ptr())}")
print(f"    -- type:  {tensor.dtype}")
print(f"    -- shape: {tensor.shape}\n")
print(tensor)

# modify PyTorch tensor
print("\nmodifying PyTorch tensor...\n")
tensor.fill_(1)
print(tensor)

# confirm changes to cudaImg
print("\nconfirming changes to cudaImage...\n")

for y in range(cuda_img.shape[0]):
    for x in range(cuda_img.shape[1]):
        print(f"cuda_img[{y}, {x}] = {cuda_img[y,x]}")
