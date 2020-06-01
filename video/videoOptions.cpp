/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "videoOptions.h"
#include <strings.h>


// constructor
videoOptions::videoOptions()
{
	width 	  = 0;
	height 	  = 0;
	frameRate   = 30;
	numBuffers  = 4;
	zeroCopy    = true;
	ioType      = INPUT;
	deviceType  = DEVICE_DEFAULT;
	flipMethod  = FLIP_DEFAULT;
	codec       = CODEC_UNKNOWN;
}


// Print
void videoOptions::Print() const
{
	printf("videoOptions\n");

	resource.Print("  ");

	printf("  -- width:      %i\n", width);
	printf("  -- height:     %i\n", width);
	printf("  -- frameRate:  %i\n", frameRate);
	printf("  -- numBuffers: %i\n", numBuffers);
	printf("  -- zeroCopy:   %i\n", (int)zeroCopy);
	printf("  -- codec:      %i\n", (int)codec);
	printf("  -- flipMethod: %i\n", (int)flipMethod);
	printf("  -- ioType:     %i\n", (int)ioType);
}


// Parse
bool videoOptions::Parse( const int argc, char** argv, videoOptions::IoType type )
{
	commandLine cmdLine(argc, argv);
	return Parse(cmdLine, type);
}


// Parse
bool videoOptions::Parse( const commandLine& cmdLine, videoOptions::IoType type )
{
	ioType = type;

	// parse input/output URI
	resource = (type == INPUT) ? cmdLine.GetString("input", "csi://0")
						  : cmdLine.GetString("output", "display://0");

	// parse stream settings
	width 	 = cmdLine.GetUnsignedInt("width");
	height 	 = cmdLine.GetUnsignedInt("height");
	frameRate  = cmdLine.GetUnsignedInt("framerate", frameRate);
	numBuffers = cmdLine.GetUnsignedInt("num-buffers", numBuffers);
	
	zeroCopy 	 = cmdLine.GetFlag("zero-copy");

	flipMethod = videoOptions::FlipMethodFromStr(cmdLine.GetString("flip-method"));
	codec 	 = videoOptions::CodecFromStr(cmdLine.GetString("codec"));
		
	return true;
}


// IoTypeToStr
const char* videoOptions::IoTypeToStr( videoOptions::IoType type )
{
	switch(type)
	{
		case INPUT:  return "input";
		case OUTPUT: return "output";
	}
}


// IoTypeFromStr
videoOptions::IoType videoOptions::IoTypeFromStr( const char* str )
{
	if( !str )
		return INPUT;

	for( int n=0; n <= OUTPUT; n++ )
	{
		const IoType value = (IoType)n;

		if( strcasecmp(str, IoTypeToStr(value)) == 0 )
			return value;
	}

	return INPUT;
}


// DeviceTypeToStr
const char* videoOptions::DeviceTypeToStr( videoOptions::DeviceType type )
{
	switch(type)
	{
		case DEVICE_DEFAULT:	return "default";
		case DEVICE_V4L2:		return "v4l2";
		case DEVICE_CSI:		return "csi";
		case DEVICE_IP:		return "ip";
		case DEVICE_FILE:		return "file";
	}
}


// DeviceTypeFromStr
videoOptions::DeviceType videoOptions::DeviceTypeFromStr( const char* str )
{
	if( !str )
		return DEVICE_DEFAULT;

	for( int n=0; n <= DEVICE_FILE; n++ )
	{
		const DeviceType value = (DeviceType)n;

		if( strcasecmp(str, DeviceTypeToStr(value)) == 0 )
			return value;
	}

	return DEVICE_DEFAULT;
}


// FlipMethodToStr
const char* videoOptions::FlipMethodToStr( videoOptions::FlipMethod flip )
{
	switch(flip)
	{
		case FLIP_NONE:				return "none";
		case FLIP_COUNTERCLOCKWISE:		return "counterclockwise";
		case FLIP_ROTATE_180:			return "rotate-180";
		case FLIP_CLOCKWISE:			return "clockwise";
		case FLIP_HORIZONTAL:			return "horizontal";
		case FLIP_UPPER_RIGHT_DIAGONAL:	return "upper-right-diagonal";
		case FLIP_VERTICAL:				return "vertical";
		case FLIP_UPPER_LEFT_DIAGONAL:	return "upper-left-diagonal";
	}
}


// FlipMethodFromStr
videoOptions::FlipMethod videoOptions::FlipMethodFromStr( const char* str )
{
	if( !str )
		return FLIP_DEFAULT;

	for( int n=0; n <= FLIP_UPPER_LEFT_DIAGONAL; n++ )
	{
		const FlipMethod value = (FlipMethod)n;

		if( strcasecmp(str, FlipMethodToStr(value)) == 0 )
			return value;
	}

	return FLIP_DEFAULT;
}


// CodecToStr
const char* videoOptions::CodecToStr( videoOptions::Codec codec )
{
	switch(codec)
	{
		case CODEC_UNKNOWN:	return "unknown";
		case CODEC_RAW:	return "raw";
		case CODEC_JPEG:	return "jpeg";
		case CODEC_H264:	return "h264";
		case CODEC_H265:	return "h265";
		case CODEC_VP8:	return "vp8";
		case CODEC_VP9:	return "vp9";
	}
}


// CodecFromStr
videoOptions::Codec videoOptions::CodecFromStr( const char* str )
{
	if( !str )
		return CODEC_UNKNOWN;

	for( int n=0; n <= CODEC_VP9; n++ )
	{
		const Codec value = (Codec)n;

		if( strcasecmp(str, CodecToStr(value)) == 0 )
			return value;
	}
	return CODEC_UNKNOWN;
}




