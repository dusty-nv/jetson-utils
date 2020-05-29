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
#include "URI.h"	


/**
 * @ingroup image
 */
struct videoOptions
{
public:
	/**
	 *
	 */
	inline videoOptions();

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
	uint32_t frameRate;
	
	/**
	 *
	 */
	uint32_t numBuffers;
	
	/**
	 *
	 */
	bool zeroCopy;

	/**
	 *
	 */
	enum DeviceType
	{
		DEVICE_DEFAULT   = 0,
		DEVICE_GSTREAMER = (1 << 0),
		DEVICE_V4L2      = (1 << 1),
		DEVICE_CSI       = (1 << 2),
		DEVICE_RTP       = (1 << 3),
		DEVICE_RTSP      = (1 << 4),
		DEVICE_FILE      = (1 << 5)
	};

	/**
	 *
	 */
	DeviceType apiType;

	/**
	 *
	 */
	DeviceType deviceType;
	
	/**
	 *
	 */
	URI resource;

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
		FLIP_DEFAULT = FLIP_ROTATE_180
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
		CODEC_JPEG,
		CODEC_H264,
		CODEC_H265,
	};

	/**
	 *
	 */
	Codec codec;


	/**
	 *
	 */
	inline void print();
};


// constructor
inline videoOptions::videoOptions()
{
	width 		= 0;
	height 		= 0;
	frameRate 	= 30;
	numBuffers  = 4;
	zeroCopy    = true;
	apiType     = DEVICE_DEFAULT;
	deviceType 	= DEVICE_DEFAULT;
	flipMethod 	= FLIP_DEFAULT;
}


// print
inline void videoOptions::print()
{
	printf("videoOptions\n");
	resource.print("  ");
	printf("  -- width:      %i\n", width);
	printf("  -- height:     %i\n", width);
	printf("  -- frameRate:  %i\n", frameRate);
	printf("  -- numBuffers: %i\n", numBuffers);
	printf("  -- zeroCopy:   %i\n", (int)zeroCopy);
	printf("  -- codec:      %i\n", (int)codec);
	printf("  -- flipMethod: %i\n", (int)flipMethod);
}

#endif

