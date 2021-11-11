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
parser.add_argument("file_out", type=str, default="images/test/cuda-example.jpg", nargs='?', help="filename of the output image to save")

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

# copy the image (this isn't necessary, just for demonstration)
copied_img = jetson.utils.cudaAllocMapped(width=input_img.width, height=input_img.height, format=input_img.format)
jetson.utils.cudaMemcpy(copied_img, input_img)   # dst, src

jetson.utils.cudaDeviceSynchronize()
jetson.utils.saveImage("images/test/memcpy_0.jpg", copied_img)
    
# or you can use this shortcut, which will allocate the image first
copied_img = jetson.utils.cudaMemcpy(input_img)

jetson.utils.cudaDeviceSynchronize()
jetson.utils.saveImage("images/test/memcpy_1.jpg", copied_img)

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

# convert back to color
color_img = convert_color(resized_img, 'rgb8')

print('color image')
print(color_img)

# draw some shapes
jetson.utils.cudaDrawCircle(color_img, (50,50), 50, (0,255,127,200)) # (cx,cy), radius, color
jetson.utils.cudaDrawRect(color_img, (200,25,350,250), (255,127,0,200)) # (left, top, right, bottom), color
jetson.utils.cudaDrawLine(color_img, (25,150), (325,15), (255,0,200,200), 10) # (x1,y1), (x2,y2), color, thickness

# save the image
if opt.file_out is not None:
	jetson.utils.cudaDeviceSynchronize()
	jetson.utils.saveImage(opt.file_out, color_img)
	print("saved {:d}x{:d} test image to '{:s}'".format(crop_img.width, crop_img.height, opt.file_out))

