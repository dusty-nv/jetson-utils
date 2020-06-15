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
 
#ifndef __VIDEO_OPTIONS_H_
#define __VIDEO_OPTIONS_H_

#include "imageFormat.h"	
#include "commandLine.h"

#include "URI.h"	


/**
 * @ingroup video
 */
struct videoOptions
{
public:
	/**
	 *
	 */
	videoOptions();
	
	/**
	 *
	 */
	URI resource;

	/**
	 *
	 */
	uint32_t width;
	
	/**
	 *
	 */
	uint32_t height;	

	/**
	 *
	 */
	float frameRate;
	
	/**
	 *
	 */
	uint32_t bitRate;

	/**
	 *
	 */
	uint32_t numBuffers;

	/**
	 *
	 */
	bool zeroCopy;
	
	/**
	 * -1 = loop forever
	 *  0 = don't loop
	 * >0 = set number of loops
	 */
	int loop;

	/**
	 *
	 */
	enum DeviceType
	{
		DEVICE_DEFAULT = 0,
		DEVICE_V4L2,
		DEVICE_CSI,
		DEVICE_IP,
		DEVICE_FILE,
		DEVICE_DISPLAY
	};

	/**
	 *
	 */
	DeviceType deviceType;

	/**
	 *
	 */
	enum IoType
	{
		INPUT = 0,
		OUTPUT,
	};

	/**
	 *
	 */
	IoType ioType;

	/**
	 * (0): none             - Identity (no rotation)
	 * (1): counterclockwise - Rotate counter-clockwise 90 degrees
	 * (2): rotate-180       - Rotate 180 degrees
	 * (3): clockwise        - Rotate clockwise 90 degrees
	 * (4): horizontal-flip  - Flip horizontally
	 * (5): upper-right-diagonal - Flip across upper right/lower left diagonal
	 * (6): vertical-flip    - Flip vertically
	 * (7): upper-left-diagonal - Flip across upper left/lower right diagonal
	 * @ingroup image
	 */
	enum FlipMethod
	{
		FLIP_NONE = 0,
		FLIP_COUNTERCLOCKWISE,
		FLIP_ROTATE_180,
		FLIP_CLOCKWISE,
		FLIP_HORIZONTAL,
		FLIP_UPPER_RIGHT_DIAGONAL,
		FLIP_VERTICAL,
		FLIP_UPPER_LEFT_DIAGONAL,
		FLIP_DEFAULT = FLIP_NONE
	};

	/**
	 *
	 */
	FlipMethod flipMethod;

	/**
	 *
	 */
	enum Codec
	{
		CODEC_UNKNOWN = 0,
		CODEC_RAW,
		CODEC_H264,
		CODEC_H265,
		CODEC_VP8,
		CODEC_VP9,
		CODEC_MPEG2,
		CODEC_MPEG4,
		CODEC_MJPEG
	};

	/**
	 *
	 */
	Codec codec;

	/**
	 *
	 */
	void Print( const char* prefix=NULL ) const;

	/**
	 *
	 */
	bool Parse( const char* URI, const int argc, char** argv, IoType ioType);

	/**
	 *
	 */
	bool Parse( const char* URI, const commandLine& cmdLine, IoType ioType);

	/**
	 *
	 */
	bool Parse( const int argc, char** argv, IoType ioType, int ioPositionArg=-1 );

	/**
	 *
	 */
	bool Parse( const commandLine& cmdLine, IoType ioType, int ioPositionArg=-1 );

	/**
	 *
	 */
	static const char* IoTypeToStr( IoType type );

	/**
	 * 
	 */
	static IoType IoTypeFromStr( const char* str );

	/**
	 *
	 */
	static const char* DeviceTypeToStr( DeviceType type );

	/**
	 * 
	 */
	static DeviceType DeviceTypeFromStr( const char* str );

	/**
	 *
	 */
	static const char* FlipMethodToStr( FlipMethod flip );

	/**
	 * 
	 */
	static FlipMethod FlipMethodFromStr( const char* str );

	/**
	 *
	 */
	static const char* CodecToStr( Codec codec );

	/**
	 * 
	 */
	static Codec CodecFromStr( const char* str );
};


/**
 * @ingroup video
 */
#define LOG_VIDEO "[video]  "

#endif

