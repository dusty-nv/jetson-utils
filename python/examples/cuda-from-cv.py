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
parser = argparse.ArgumentParser(description='Convert an image from OpenCV to CUDA')

parser.add_argument("file_in", type=str, default="images/granny_smith_1.jpg", nargs='?', help="filename of the input image to process")
parser.add_argument("file_out", type=str, default="cuda-from-cv.jpg", nargs='?', help="filename of the output image to save")

opt = parser.parse_args()


# load the image
cv_img = cv2.imread(opt.file_in)

print('OpenCV image size: ' + str(cv_img.shape))
print('OpenCV image type: ' + str(cv_img.dtype))

# convert to CUDA (cv2 images are numpy arrays, in BGR format)
bgr_img = jetson.utils.cudaFromNumpy(cv_img, isBGR=True)

print('BGR image: ')
print(bgr_img)

# convert from BGR -> RGB
rgb_img = jetson.utils.cudaAllocMapped(width=bgr_img.width,
							    height=bgr_img.height,
							    format='rgb8')

jetson.utils.cudaConvertColor(bgr_img, rgb_img)

print('RGB image: ')
print(rgb_img)

# save the image
if opt.file_out is not None:
	jetson.utils.cudaDeviceSynchronize()
	jetson.utils.saveImage(opt.file_out, rgb_img)
	print("saved {:d}x{:d} test image to '{:s}'".format(rgb_img.width, rgb_img.height, opt.file_out))

