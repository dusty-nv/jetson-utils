#!/usr/bin/python3
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
import numpy as np

# parse the command line
parser = argparse.ArgumentParser(description='Copy a test image from numpy to CUDA and save it to disk')

parser.add_argument("--width", type=int, default=512, help="width of the array (in pixels)")
parser.add_argument("--height", type=int, default=256, help="height of the array (in pixels)")
parser.add_argument("--depth", type=int, default=4, help="number of color channels in the array (1, 3, or 4)")
parser.add_argument("--dtype", type=str, default="float32", help="numpy data type: " + " | ".join(sorted({str(key) for key in np.sctypeDict.keys()})))
parser.add_argument("--filename", type=str, default="cuda-from-numpy.jpg", help="filename of the output test image")

opt = parser.parse_args()


# create numpy ndarray
array = np.ndarray(shape=(opt.height, opt.width, opt.depth), dtype=np.dtype(opt.dtype))

print('numpy array shape:  ' + str(array.shape))
print('numpy array dtype:  ' + str(array.dtype))

# fill array with test colors
for y in range(opt.height):
	for x in range(opt.width):
		px = [ 0, float(x) / float(opt.width) * 255, float(y) / float(opt.height) * 255, 255]
		
		if opt.depth == 1:
			px = [ px[0] * 0.2989 + px[1] * 0.5870 + px[2] * 0.1140 ];
		elif opt.depth == 3:
			px.pop()

		array[y, x] = px

# copy to CUDA memory
cuda_mem = jetson.utils.cudaFromNumpy(array)
print(cuda_mem)

# save as image
jetson.utils.saveImageRGBA(opt.filename, cuda_mem, opt.width, opt.height)
print("saved {:d}x{:d} test image to '{:s}'".format(opt.width, opt.height, opt.filename))

