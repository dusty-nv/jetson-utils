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

#include "logging.h"
#include <strings.h>


// constructor
videoOptions::videoOptions()
{
	width 	  = 0;
	height 	  = 0;
	frameRate   = 0;
	bitRate     = 0;
	numBuffers  = 4;
	loop        = 0;
	rtspLatency = 2000;
	zeroCopy    = true;
	ioType      = INPUT;
	deviceType  = DEVICE_DEFAULT;
	flipMethod  = FLIP_DEFAULT;
	codec       = CODEC_UNKNOWN;
}


// Print
void videoOptions::Print( const char* prefix ) const
{
	LogInfo("------------------------------------------------\n");

	if( prefix != NULL )
		LogInfo("%s video options:\n", prefix);
	else
		LogInfo("video options:\n");

	LogInfo("------------------------------------------------\n");
	resource.Print("  ");

	LogInfo("  -- deviceType: %s\n", DeviceTypeToStr(deviceType));
	LogInfo("  -- ioType:     %s\n", IoTypeToStr(ioType));
	LogInfo("  -- codec:      %s\n", CodecToStr(codec));
	LogInfo("  -- width:      %u\n", width);
	LogInfo("  -- height:     %u\n", height);
	LogInfo("  -- frameRate:  %f\n", frameRate);
	LogInfo("  -- bitRate:    %u\n", bitRate);
	LogInfo("  -- numBuffers: %u\n", numBuffers);
	LogInfo("  -- zeroCopy:   %s\n", zeroCopy ? "true" : "false");	
	LogInfo("  -- flipMethod: %s\n", FlipMethodToStr(flipMethod));
	LogInfo("  -- loop:       %i\n", loop);
	LogInfo("  -- rtspLatency %i\n", rtspLatency);
	
	LogInfo("------------------------------------------------\n");
}


// Parse
bool videoOptions::Parse( const char* URI, const int argc, char** argv, videoOptions::IoType type, const char* extraFlag )
{
	commandLine cmdLine(argc, argv, extraFlag);

	return Parse(URI, cmdLine, type);
}


// Parse
bool videoOptions::Parse( const char* URI, const commandLine& cmdLine, videoOptions::IoType type )
{
	ioType = type;

	// check for headless mode
	const bool headless = cmdLine.GetFlag("no-display") | cmdLine.GetFlag("headless");

	if( (URI == NULL || strlen(URI) == 0) && type == OUTPUT && !headless )
		URI = "display://0";

	// parse input/output URI
	if( !resource.Parse(URI) )
	{
		LogError(LOG_VIDEO "videoOptions -- failed to parse %s resource URI (%s)\n", IoTypeToStr(type), URI != NULL ? URI : "null");
		return false;
	}

	// parse stream settings
	numBuffers = cmdLine.GetUnsignedInt("num-buffers", numBuffers);
	//zeroCopy = cmdLine.GetFlag("zero-copy");	// no default returned, so disable this for now

	// width
	width = (type == INPUT) ? cmdLine.GetUnsignedInt("input-width")
					    : cmdLine.GetUnsignedInt("output-width");

	if( width == 0 )
		width = cmdLine.GetUnsignedInt("width");

	// height
	height = (type == INPUT) ? cmdLine.GetUnsignedInt("input-height")
					     : cmdLine.GetUnsignedInt("output-height");

	if( height == 0 )
		height = cmdLine.GetUnsignedInt("height");

	// framerate
	frameRate = (type == INPUT) ? cmdLine.GetFloat("input-rate", frameRate)
						   : cmdLine.GetFloat("output-rate", frameRate);

	if( frameRate == 0 )
		frameRate = cmdLine.GetFloat("framerate");

	// flip-method
	const char* flipStr = (type == INPUT) ? cmdLine.GetString("input-flip")
								   : cmdLine.GetString("output-flip");

	if( !flipStr )
	{
		flipStr = (type == INPUT) ? cmdLine.GetString("input-flip-method")
							 : cmdLine.GetString("output-flip-method");

		if( !flipStr )
			flipStr = cmdLine.GetString("flip-method");
	}
	
	flipMethod = videoOptions::FlipMethodFromStr(flipStr);

	// codec
	const char* codecStr = (type == INPUT) ? cmdLine.GetString("input-codec")
								    : cmdLine.GetString("output-codec");

	if( !codecStr && type == OUTPUT )
		codecStr = cmdLine.GetString("codec");

	if( codecStr != NULL )	
		codec = videoOptions::CodecFromStr(codecStr);
		
	// bitrate
	if( type == OUTPUT )
		bitRate = cmdLine.GetUnsignedInt("bitrate", bitRate);

	// loop
	if( type == INPUT )
	{
		loop = cmdLine.GetInt("input-loop", -999);
		
		if( loop == -999 )
			loop = cmdLine.GetInt("loop");
	}

	// RTSP latency
	rtspLatency = cmdLine.GetUnsignedInt("input-rtsp-latency", rtspLatency);
	
	return true;
}


// Parse
bool videoOptions::Parse( const commandLine& cmdLine, videoOptions::IoType type, int ioPositionArg )
{
	// check for headless output
	const bool headless = cmdLine.GetFlag("no-display") | cmdLine.GetFlag("headless");

	// parse input/output URI
	const char* resourceStr = NULL;

	if( ioPositionArg >= 0 && ioPositionArg < cmdLine.GetPositionArgs() )
		resourceStr = cmdLine.GetPosition(ioPositionArg);

	if( !resourceStr )
	{
		if( type == INPUT )
		{
			resourceStr = cmdLine.GetString("camera", "csi://0");

			//if( !resourceStr )
			//	resourceStr = cmdLine.GetString("input", "csi://0");
		}
		else
		{
			//resourceStr = cmdLine.GetString("output", headless ? NULL : "display://0");	// BUG:  "output" will return flags with longer names, i.e. "output-blob"

			if( !headless )
				resourceStr = "display://0";
		}
	}

	return Parse(resourceStr, cmdLine, type);
}


// Parse
bool videoOptions::Parse( const int argc, char** argv, videoOptions::IoType type, int ioPositionArg )
{
	commandLine cmdLine(argc, argv);
	return Parse(cmdLine, type, ioPositionArg);
}



// IoTypeToStr
const char* videoOptions::IoTypeToStr( videoOptions::IoType type )
{
	switch(type)
	{
		case INPUT:  return "input";
		case OUTPUT: return "output";
	}
	return nullptr;
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
		case DEVICE_DISPLAY:	return "display";
	}
	return nullptr;
}


// DeviceTypeFromStr
videoOptions::DeviceType videoOptions::DeviceTypeFromStr( const char* str )
{
	if( !str )
		return DEVICE_DEFAULT;

	for( int n=0; n <= DEVICE_DISPLAY; n++ )
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
	return nullptr;
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
		case CODEC_H264:	return "h264";
		case CODEC_H265:	return "h265";
		case CODEC_VP8:	return "vp8";
		case CODEC_VP9:	return "vp9";
		case CODEC_MPEG2:	return "mpeg2";
		case CODEC_MPEG4:	return "mpeg4";
		case CODEC_MJPEG:	return "mjpeg";
	}
	return nullptr;
}


// CodecFromStr
videoOptions::Codec videoOptions::CodecFromStr( const char* str )
{
	if( !str )
		return CODEC_UNKNOWN;

	for( int n=0; n <= CODEC_MJPEG; n++ )
	{
		const Codec value = (Codec)n;

		if( strcasecmp(str, CodecToStr(value)) == 0 )
			return value;
	}
	return CODEC_UNKNOWN;
}




