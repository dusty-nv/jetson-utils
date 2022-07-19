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
import numpy

from jetson_utils import cudaAllocMapped, cudaToNumpy

# parse the command line
parser = argparse.ArgumentParser('Map CUDA to memory to numpy ndarray')

parser.add_argument("--width", type=int, default=4, help="width of the array (in float elements)")
parser.add_argument("--height", type=int, default=2, help="height of the array (in float elements)")

opt = parser.parse_args()

# allocate cuda memory
cuda_img = cudaAllocMapped(width=opt.width, height=opt.height, format='rgb32f')
print(cuda_img)

# create a numpy ndarray that references the CUDA memory
# it won't be copied, but uses the same memory underneath
array = cudaToNumpy(cuda_img)

print("\ncudaToNumpy() array:")
print(type(array))
print(array.dtype)
print(array.shape)	# numpy dims will be in (height, width, depth) order
print(array)

# create another ndarray and do some ops with it
array2 = numpy.ones(array.shape, numpy.float32)

print("\nadding arrays...\n")
print(array + array2)


