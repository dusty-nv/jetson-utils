#!/usr/bin/env python3
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

import sys
import argparse

from jetson_utils import videoSource, videoOutput, Log

# parse command line
parser = argparse.ArgumentParser(description="View various types of video streams", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)    # default:  options={'width': 1280, 'height': 720, 'framerate': 30}
output = videoOutput(args.output, argv=sys.argv)  # default:  options={'width': 1280, 'height': 720, 'framerate': 30}

# capture frames until EOS or user exits
numFrames = 0

while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  
        
    if numFrames % 25 == 0 or numFrames < 15:
        Log.Verbose(f"video-viewer:  captured {numFrames} frames ({img.width} x {img.height})")
	
    numFrames += 1
	
    # render the image
    output.Render(img)
    
    # update the title bar
    output.SetStatus("Video Viewer | {:d}x{:d} | {:.1f} FPS".format(img.width, img.height, output.GetFrameRate()))
	
    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

