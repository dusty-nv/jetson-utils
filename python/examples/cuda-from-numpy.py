#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import jetson.utils
import argparse
import numpy

# parse the command line
parser = argparse.ArgumentParser(description='Copy a test image from numpy to CUDA and save it to disk')

parser.add_argument("--width", type=int, default=512, help="width of the array (in float elements)")
parser.add_argument("--height", type=int, default=256, help="height of the array (in float elements)")
parser.add_argument("--depth", type=int, default=4, help="depth of the array (in float elements)")
parser.add_argument("--filename", type=str, default="cuda-from-numpy.jpg", help="filename of the output test image")

opt = parser.parse_args()

# create numpy ndarray
array = numpy.ndarray(shape=(opt.height, opt.width, opt.depth))

# fill array with test colors
for y in range(opt.height):
	for x in range(opt.width):
		array[y, x] = [ 0, float(x) / float(opt.width) * 255, float(y) / float(opt.height) * 255, 255]

# copy to CUDA memory
cuda_mem = jetson.utils.cudaFromNumpy(array)
print(cuda_mem)

# save as image
jetson.utils.saveImageRGBA(opt.filename, cuda_mem, opt.width, opt.height)
print("saved {:d}x{:d} test image to '{:s}'".format(opt.width, opt.height, opt.filename))

