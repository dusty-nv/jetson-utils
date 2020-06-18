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

# parse the command line
parser = argparse.ArgumentParser()

parser.add_argument("--title", type=str, default="Test OpenGL Display Window", help="desired title string of window")
parser.add_argument("--r", type=float, default=0.00, help="window background color (Red component, 0.0-1.0)")
parser.add_argument("--g", type=float, default=0.75, help="window background color (Green component, 0.0-1.0)")
parser.add_argument("--b", type=float, default=0.25, help="window background color (Blue component, 0.0-1.0)")
parser.add_argument("--a", type=float, default=0.00, help="window background color (Alpha component, 0.0-1.0)")

opt = parser.parse_args()
print(opt)

# create display device
display = jetson.utils.glDisplay(opt.title, opt.r, opt.g, opt.b, opt.a)

# render until user exits
while display.IsOpen():
	display.BeginRender()
	display.EndRender()
	display.SetTitle("{:s} | {:.0f} FPS".format(opt.title, display.GetFPS()))


