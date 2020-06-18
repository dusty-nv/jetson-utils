#!/usr/bin/python3
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

import cv2
import jetson.utils
import argparse


# parse the command line
parser = argparse.ArgumentParser(description='Convert an image from CUDA to OpenCV')

parser.add_argument("file_in", type=str, default="images/jellyfish.jpg", nargs='?', help="filename of the input image to process")
parser.add_argument("file_out", type=str, default="cuda-to-cv.jpg", nargs='?', help="filename of the output image to save")

opt = parser.parse_args()


# load the image into CUDA memory
rgb_img = jetson.utils.loadImage(opt.file_in)

print('RGB image: ')
print(rgb_img)

# convert to BGR, since that's what OpenCV expects
bgr_img = jetson.utils.cudaAllocMapped(width=rgb_img.width,
							    height=rgb_img.height,
							    format='bgr8')

jetson.utils.cudaConvertColor(rgb_img, bgr_img)

print('BGR image: ')
print(bgr_img)

# make sure the GPU is done work before we convert to cv2
jetson.utils.cudaDeviceSynchronize()

# convert to cv2 image (cv2 images are numpy arrays)
cv_img = jetson.utils.cudaToNumpy(bgr_img)

print('OpenCV image size: ' + str(cv_img.shape))
print('OpenCV image type: ' + str(cv_img.dtype))

# save the image
if opt.file_out is not None:
	cv2.imwrite(opt.file_out, cv_img)
	print("saved {:d}x{:d} test image to '{:s}'".format(bgr_img.width, bgr_img.height, opt.file_out))

