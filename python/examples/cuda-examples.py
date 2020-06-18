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

import jetson.utils
import argparse


# parse the command line
parser = argparse.ArgumentParser(description='Perform some example CUDA processing on an image')

parser.add_argument("file_in", type=str, default="images/granny_smith_1.jpg", nargs='?', help="filename of the input image to process")
parser.add_argument("file_out", type=str, default="cuda-example.jpg", nargs='?', help="filename of the output image to save")

opt = parser.parse_args()


# convert colorspace
def convert_color(img, output_format):
	converted_img = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format=output_format)
	jetson.utils.cudaConvertColor(img, converted_img)
	return converted_img


# center crop an image
def crop(img, crop_factor):
	crop_border = ((1.0 - crop_factor[0]) * 0.5 * img.width,
				(1.0 - crop_factor[1]) * 0.5 * img.height)

	crop_roi = (crop_border[0], crop_border[1], img.width - crop_border[0], img.height - crop_border[1])

	crop_img = jetson.utils.cudaAllocMapped(width=img.width * crop_factor[0],
									height=img.height * crop_factor[1],
									format=img.format)

	jetson.utils.cudaCrop(img, crop_img, crop_roi)
	return crop_img


# resize an image
def resize(img, resize_factor):
	resized_img = jetson.utils.cudaAllocMapped(width=img.width * resize_factor[0],
									   height=img.height * resize_factor[1],
									   format=img.format)

	jetson.utils.cudaResize(img, resized_img)
	return resized_img


# load the image
input_img = jetson.utils.loadImage(opt.file_in)

print('input image:')
print(input_img)

# convert to grayscale - other formats are:
#  rgb8, rgba8, rgb32f, rgba32f, gray32f
gray_img = convert_color(input_img, "gray8")

print('grayscale image:')
print(gray_img)

# crop the image
crop_img = crop(gray_img, (0.75, 0.75))

print('cropped image:')
print(crop_img)

# resize the image
resized_img = resize(crop_img, (0.5, 0.5))

print('resized image:')
print(resized_img)

# save the image
if opt.file_out is not None:
	jetson.utils.cudaDeviceSynchronize()
	jetson.utils.saveImage(opt.file_out, resized_img)
	print("saved {:d}x{:d} test image to '{:s}'".format(crop_img.width, crop_img.height, opt.file_out))

