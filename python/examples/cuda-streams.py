#!/usr/bin/env python3
# Test multiple CUDA streams
import sys
import argparse

from jetson_utils import (videoSource, videoOutput, loadImage, saveImage, Log,
                          cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize,
                          cudaStreamWaitEvent, cudaMemcpy, cudaMalloc, cudaMallocMapped,
                          cudaEventCreate, cudaEventDestroy, cudaEventRecord, cudaResize)

parser = argparse.ArgumentParser(description="View various types of video streams", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("--image-in", type=str, default="images/granny_smith_1.jpg", nargs='?', help="filename of the input image to process")
parser.add_argument("--image-out", type=str, default="images/test/cuda-streams.jpg", nargs='?', help="filename of the output image to save")

parser.add_argument("--video-input", type=str, default=None, help="URI of the input stream")
parser.add_argument("--video-output", type=str, default=None, help="URI of the output stream")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# 
# load an image and test some operations on the stream
#
stream = cudaStreamCreate(nonblocking=True)
stream_blocking = cudaStreamCreate()

print('cuda stream (non-blocking)', stream)
print('cuda stream (blocking)    ', stream_blocking)


img_in = loadImage(args.image_in, stream=stream)

print('input image\n', img_in)

print('cudaMalloc()\n', cudaMalloc(like=img_in))
print('cudaMemcpy()\n', cudaMemcpy(img_in, stream=stream))

img_out = cudaMallocMapped(
    width=img_in.width*0.5,
    height=img_in.height*0.5,
    format=img_in.format
)

print('img_out\n', img_out)

cudaResize(img_in, img_out, stream=stream)

saveImage(args.image_out, img_out, stream=stream)


# 
# run an optional video test, where the input and outputs
# operate on different CUDA streams with synchronization
#
if not args.video_input or not args.video_output:
    print("--video-input and --video-output not specified, skipping test of video streams")
    sys.exit(0)
    
input = videoSource(args.video_input, argv=sys.argv)
output = videoOutput(args.video_output, argv=sys.argv)

input_stream = cudaStreamCreate()
output_stream = cudaStreamCreate()

# capture frames until EOS or user exits
numFrames = 0

while True:
    # capture the next image
    img = input.Capture(stream=input_stream)
    img.stream = input_stream
    img.event = cudaEventRecord(stream=input_stream) #cudaEventCreate()
    print(img)
    
    if img is None: # timeout
        continue  
        
    if numFrames % 25 == 0 or numFrames < 15:
        Log.Verbose(f"video-viewer:  captured {numFrames} frames ({img.width} x {img.height})")
	
    numFrames += 1
	
    # render the image
    cudaStreamWaitEvent(output_stream, img.event)
    output.Render(img, stream=output_stream)
    
    img.stream = None
    img.event = None # cudaEventDestroy()
    
    # update the title bar
    output.SetStatus("Video Viewer | {:d}x{:d} | {:.1f} FPS".format(img.width, img.height, output.GetFrameRate()))
	
    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

