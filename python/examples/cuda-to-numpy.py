#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np

from jetson_utils import cudaImage, cudaToNumpy

# parse the command line
parser = argparse.ArgumentParser('Map CUDA to memory to numpy ndarray')

parser.add_argument("--width", type=int, default=4, help="width of the array (in float elements)")
parser.add_argument("--height", type=int, default=2, help="height of the array (in float elements)")

args = parser.parse_args()

# allocate cuda memory
cuda_img = cudaImage(width=args.width, height=args.height, format='rgb32f')

print(cuda_img)

# create a numpy array and do some ops with it
array = np.ones(cuda_img.shape, np.float32)

# since cudaImage supports __array__ interface, we can do numpy ops on it directly
print(np.add(cuda_img, array)) 

# explicitly create a numpy array that references the same CUDA memory (it won't be copied)
# (this typically isn't necessary to use anymore with cudaImage's __array__ interface)
mapped_array = cudaToNumpy(cuda_img)

print("\ncudaToNumpy() array:")
print(type(mapped_array))
print(f"   -- ptr:   {hex(mapped_array.ctypes.data)}")
print(f"   -- type:  {mapped_array.dtype}")
print(f"   -- shape: {mapped_array.shape}\n") # numpy dims will be in (height, width, depth) order
print(mapped_array)                           # this should print out one's

print("\nadding arrays...\n")
print(array + mapped_array)  # this should print out ones's


