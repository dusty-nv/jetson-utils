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
#include "gstUtility.h"

#include "logging.h"
#include <strings.h>


// constructor
videoOptions::videoOptions()
{
	width 	  = 0;
	height 	  = 0;
	frameRate   = 0;
	frameCount  = 0;
	bitRate     = 0;
	numBuffers  = 4;
	loop        = 0;
	latency     = 10;
	zeroCopy    = true;
	ioType      = INPUT;
	deviceType  = DEVICE_DEFAULT;
	flipMethod  = FLIP_DEFAULT;
	codec       = CODEC_UNKNOWN;
	codecType   = gst_default_codec();
}


// Print
void videoOptions::Print( const char* prefix ) const
{
	LogInfo("------------------------------------------------\n");

	if( prefix != NULL ) {
		LogInfo("%s video options:\n", prefix);
  }
	else {
		LogInfo("video options:\n");
  }

	LogInfo("------------------------------------------------\n");
	resource.Print("  ");

	LogInfo("  -- deviceType: %s\n", DeviceTypeToStr(deviceType));
	LogInfo("  -- ioType:     %s\n", IoTypeToStr(ioType));
	
	if( save.path.length() > 0 )
		LogInfo("  -- save:       %s\n", save.path.c_str());

	if( deviceType != DEVICE_CSI && deviceType != DEVICE_DISPLAY )
	{
		LogInfo("  -- codec:      %s\n", CodecToStr(codec));
		LogInfo("  -- codecType:  %s\n", CodecTypeToStr(codecType));
	}
	
	if( width != 0 )
		LogInfo("  -- width:      %u\n", width);
	
	if( height != 0 )
		LogInfo("  -- height:     %u\n", height);
	
	LogInfo("  -- frameRate:  %g\n", frameRate);
	
	if( ioType == OUTPUT && (deviceType == DEVICE_IP || deviceType == DEVICE_FILE) )
		LogInfo("  -- bitRate:    %u\n", bitRate);
	
	LogInfo("  -- numBuffers: %u\n", numBuffers);
	LogInfo("  -- zeroCopy:   %s\n", zeroCopy ? "true" : "false");	
	
	if( ioType == INPUT )
	{
		LogInfo("  -- flipMethod: %s\n", FlipMethodToStr(flipMethod));
	
		if( deviceType != DEVICE_CSI && deviceType != DEVICE_V4L2 )
			LogInfo("  -- loop:       %i\n", loop);
	}
	
	if( deviceType == DEVICE_IP )
		LogInfo("  -- latency     %i\n", latency);
	
	if( stunServer.length() > 0 )
		LogInfo("  -- stunServer  %s\n", stunServer.c_str());

	if( sslCert.length() > 0 )
		LogInfo("  -- sslCert     %s\n", sslCert.c_str());
	
	if( sslKey.length() > 0 )
		LogInfo("  -- sslKey      %s\n", sslKey.c_str());
	
	LogInfo("------------------------------------------------\n");
}

#define VALID_STR(x) (x != NULL && strlen(x) > 0)

// Parse
bool videoOptions::Parse( const char* URI, const commandLine& cmdLine, videoOptions::IoType type, int ioPositionArg )
{
	ioType = type;

	// check for headless mode
	const bool headless = cmdLine.GetFlag("no-display") | cmdLine.GetFlag("headless");

	// check for positional args
	if( ioPositionArg >= 0 && ioPositionArg < cmdLine.GetPositionArgs() )
		URI = cmdLine.GetPosition(ioPositionArg);
	
	// default URI's
	if( !VALID_STR(URI) && !resource.valid() ) 
	{
		if( type == INPUT )
			URI = "csi://0";
		else if( type == OUTPUT && !headless )
			URI = "display://0";
	}
	
	// parse input/output URI
	if( VALID_STR(URI) )
	{
		if( !resource.Parse(URI) )
		{
			LogError(LOG_VIDEO "videoOptions -- failed to parse %s resource URI (%s)\n", IoTypeToStr(type), URI != NULL ? URI : "null");
			return false;
		}
	}
	else if( !resource.valid() )
	{
		LogError(LOG_VIDEO "videoOptions -- %s resource URI was not set with a valid string\n", IoTypeToStr(type));
		return false;
	}
	
	deviceType = DeviceTypeFromStr(resource.protocol.c_str());
	
	// parse 'save' URI
	const char* save_path = (type == INPUT) ? cmdLine.GetString("input-save")
	                                        : cmdLine.GetString("output-save");
								
	if( save_path != NULL )
	{
		if( !save.Parse(save_path) )
		{
			LogError(LOG_VIDEO "videoOptions -- failed to parse --%s-save path (%s)\n", IoTypeToStr(type), save_path);
			return false;
		}
		
		if( DeviceTypeFromStr(save.protocol.c_str()) != DEVICE_FILE )
		{
			LogError(LOG_VIDEO "videoOptions -- the --%s-save argument must be a file path (%s)\n", IoTypeToStr(type), save_path);
			return false;
		}
	}
	
	// parse stream settings
	numBuffers = cmdLine.GetUnsignedInt("num-buffers", numBuffers);
	//zeroCopy = cmdLine.GetFlag("zero-copy");	// no default returned, so disable this for now

	// width
	width = (type == INPUT) ? cmdLine.GetUnsignedInt("input-width", width)
					    : cmdLine.GetUnsignedInt("output-width", width);

	if( width == 0 )
		width = cmdLine.GetUnsignedInt("width");

	// height
	height = (type == INPUT) ? cmdLine.GetUnsignedInt("input-height", height)
					     : cmdLine.GetUnsignedInt("output-height", height);

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
	
	if( flipStr != NULL )
		flipMethod = videoOptions::FlipMethodFromStr(flipStr);

	// codec
	const char* codecStr = (type == INPUT) ? cmdLine.GetString("input-codec")
								    : cmdLine.GetString("output-codec");

	if( !codecStr && type == OUTPUT )
		codecStr = cmdLine.GetString("codec");

	if( codecStr != NULL )	
		codec = videoOptions::CodecFromStr(codecStr);
		
	// codec type
	const char* codecTypeStr = (type == INPUT) ? cmdLine.GetString("input-decoder")
								        : cmdLine.GetString("output-encoder");
									   
	if( codecTypeStr != NULL )	
		codecType = videoOptions::CodecTypeFromStr(codecTypeStr);

	// bitrate
	if( type == OUTPUT )
		bitRate = cmdLine.GetUnsignedInt("bitrate", bitRate);

	// loop
	if( type == INPUT )
		loop = cmdLine.GetInt("input-loop", cmdLine.GetInt("loop", loop));

	// latency
	latency = (type == INPUT) ? cmdLine.GetUnsignedInt("input-latency", cmdLine.GetUnsignedInt("input-rtsp-latency", latency))
						 : cmdLine.GetUnsignedInt("output-latency", latency);
	
	// STUN server
	const char* stunStr = cmdLine.GetString("stun-server");
	
	if( stunStr != NULL )
		stunServer = stunStr;

	// SSL certificate/key
	const char* keyStr = cmdLine.GetString("ssl-key", getenv("SSL_KEY"));
	const char* certStr = cmdLine.GetString("ssl-cert", getenv("SSL_CERT"));
	
	if( keyStr )
		sslKey = keyStr;
	
	if( certStr )
		sslCert = certStr;

	return true;
}


// Parse
bool videoOptions::Parse( const char* URI, const int argc, char** argv, videoOptions::IoType type, int ioPositionArg )
{
	return Parse(URI, commandLine(argc, argv), type, ioPositionArg);
}


// Parse
bool videoOptions::Parse( const int argc, char** argv, videoOptions::IoType type, int ioPositionArg )
{
	return Parse(commandLine(argc, argv), type, ioPositionArg);
}


// Parse
bool videoOptions::Parse( const commandLine& cmdLine, videoOptions::IoType type, int ioPositionArg )
{
	return Parse(NULL, cmdLine, type, ioPositionArg);
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

	if( strcasecmp(str, "rtp") == 0 || strcasecmp(str, "rtsp") == 0 || strcasecmp(str, "rtmp") == 0 || strcasecmp(str, "rtpmp2ts") == 0 || strcasecmp(str, "webrtc") == 0 )
		return DEVICE_IP;

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
		case FLIP_HORIZONTAL:			return "horizontal-flip";
		case FLIP_UPPER_RIGHT_DIAGONAL:	return "upper-right-diagonal";
		case FLIP_VERTICAL:				return "vertical-flip";
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
	
	if( strcasecmp(str, "horizontal") == 0 )
		return FLIP_HORIZONTAL;
	else if( strcasecmp(str, "vertical") == 0 )
		return FLIP_VERTICAL;

	return FLIP_DEFAULT;
}


// CodecToStr
const char* videoOptions::CodecToStr( videoOptions::Codec codec )
{
	switch(codec)
	{
		case CODEC_UNKNOWN:	return "unknown";
		case CODEC_RAW:	return "raw";
		case CODEC_H264:	return "H264";
		case CODEC_H265:	return "H265";
		case CODEC_VP8:	return "VP8";
		case CODEC_VP9:	return "VP9";
		case CODEC_MPEG2:	return "MPEG2";
		case CODEC_MPEG4:	return "MPEG4";
		case CODEC_MJPEG:	return "MJPEG";
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


// CodecTypeToStr
const char* videoOptions::CodecTypeToStr( videoOptions::CodecType codec )
{
	switch(codec)
	{
		case CODEC_CPU:   return "cpu";
		case CODEC_OMX:   return "omx";
		case CODEC_V4L2:  return "v4l2";
		case CODEC_NVENC: return "nvenc";
		case CODEC_NVDEC: return "nvdec";
	}
	
	return nullptr;
}


// CodecFromStr
videoOptions::CodecType videoOptions::CodecTypeFromStr( const char* str )
{
	if( !str )
		return gst_default_codec();

	for( int n=0; n <= CODEC_NVDEC; n++ )
	{
		const CodecType value = (CodecType)n;

		if( strcasecmp(str, CodecTypeToStr(value)) == 0 )
			return value;
	}
	
	return gst_default_codec();
}

