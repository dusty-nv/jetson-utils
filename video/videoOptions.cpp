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
	bitRate     = 0;
	numBuffers  = 4;
	zeroCopy    = true;
	ioType      = INPUT;
	deviceType  = DEVICE_DEFAULT;
	flipMethod  = FLIP_DEFAULT;
	codec       = CODEC_UNKNOWN;
}


// Print
void videoOptions::Print( const char* prefix ) const
{
	printf("------------------------------------------------\n");

	if( prefix != NULL )
		printf("%s video options:\n", prefix);
	else
		printf("video options:\n");

	printf("------------------------------------------------\n");
	resource.Print("  ");

	printf("  -- width:      %u\n", width);
	printf("  -- height:     %u\n", height);
	printf("  -- frameRate:  %u\n", frameRate);
	printf("  -- bitRate:    %u\n", bitRate);
	printf("  -- numBuffers: %u\n", numBuffers);
	printf("  -- zeroCopy:   %s\n", zeroCopy ? "true" : "false");
	printf("  -- codec:      %s\n", CodecToStr(codec));
	printf("  -- flipMethod: %s\n", FlipMethodToStr(flipMethod));
	printf("  -- ioType:     %s\n", IoTypeToStr(ioType));

	printf("------------------------------------------------\n");
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

	// check for headless output
	const bool headless = cmdLine.GetFlag("no-display") | cmdLine.GetFlag("headless");

	// parse input/output URI
	const char* resourceStr = (type == INPUT) ? cmdLine.GetString("input", "csi://0")
						                 : cmdLine.GetString("output", headless ? NULL : "display://0");

	if( !resource.Parse(resourceStr) )
	{
		printf("videoOptions -- failed to parse %s resource URI (%s)\n", IoTypeToStr(type), resourceStr != NULL ? resourceStr : "null");
		return false;
	}

	// parse stream settings
	width 	 = cmdLine.GetUnsignedInt("width");
	height 	 = cmdLine.GetUnsignedInt("height");
	frameRate  = cmdLine.GetUnsignedInt("framerate", frameRate);
	numBuffers = cmdLine.GetUnsignedInt("num-buffers", numBuffers);
	zeroCopy 	 = cmdLine.GetFlag("zero-copy");

	// flip-method
	const char* flipStr = (type == INPUT) ? cmdLine.GetString("input-flip-method")
								   : cmdLine.GetString("output-flip-method");

	if( !flipStr )
		flipStr = cmdLine.GetString("flip-method");

	flipMethod = videoOptions::FlipMethodFromStr(flipStr);

	// codec
	const char* codecStr = (type == INPUT) ? cmdLine.GetString("input-codec")
								    : cmdLine.GetString("output-codec");

	if( !codecStr )
		codecStr = cmdLine.GetString("codec");

	codec = videoOptions::CodecFromStr(codecStr);
		
	// bitrate
	if( type == OUTPUT )
		bitRate = cmdLine.GetUnsignedInt("bitrate", bitRate);

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
		case CODEC_MPEG2:	return "mpeg2";
		case CODEC_MPEG4:	return "mpeg4";
	}
}


// CodecFromStr
videoOptions::Codec videoOptions::CodecFromStr( const char* str )
{
	if( !str )
		return CODEC_UNKNOWN;

	for( int n=0; n <= CODEC_MPEG4; n++ )
	{
		const Codec value = (Codec)n;

		if( strcasecmp(str, CodecToStr(value)) == 0 )
			return value;
	}
	return CODEC_UNKNOWN;
}




