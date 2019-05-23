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


# parse the command line
parser = argparse.ArgumentParser()

parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--title", type=str, default="Camera Viewer", help="default title of display window")
parser.add_argument("--v4l2_device", type=int, default=-1, help="if using VL42 camera, index of the desired /dev/video node")

opt = parser.parse_args()
print(opt)

# create display window
display = jetson.utils.glDisplay(opt.title)

# create camera device
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.v4l2_device)

# open the camera for streaming
camera.Open()

# capture frames until user exits
while display.IsOpen():
	image, width, height = camera.CaptureRGBA()
	display.RenderOnce(image, width, height)
	display.SetTitle("{:s} | {:d}x{:d} | {:f} FPS".format(opt.title, width, height, display.GetFPS()))
	
# close the camera
camera.Close()

