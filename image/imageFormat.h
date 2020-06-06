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
 
#ifndef __IMAGE_FORMAT_H_
#define __IMAGE_FORMAT_H_


// include vector types (float4, float3, uchar4, uchar3, ect.)
#include "cudaUtility.h"		


/**
 * @ingroup image
 */
enum imageFormat
{
	// RGB
	IMAGE_RGB8=0,
	IMAGE_RGB32F,

	// RGBA
	IMAGE_RGBA8,
	IMAGE_RGBA32F,

	// YUV
	IMAGE_YUYV,
	IMAGE_YUY2=IMAGE_YUYV,
	IMAGE_UYVY,		
	IMAGE_I420,
	IMAGE_YV12,
	IMAGE_NV12,
	
	// grayscale
	IMAGE_GRAY8,	
	IMAGE_GRAY32,

	// extras
	IMAGE_COUNT,
	IMAGE_UNKNOWN=999,
	IMAGE_DEFAULT=IMAGE_RGBA32F
};

/**
 * @ingroup image
 */
inline const char* imageFormatToStr( imageFormat format );

/**
 * @ingroup image
 */
inline imageFormat imageFormatFromStr( const char* str );

/**
 * Number of image channels in format
 * @ingroup image
 */
inline size_t imageFormatChannels( imageFormat format );

/**
 * Pixel bit depth (in bits, not bytes)
 * @ingroup image
 */
inline size_t imageFormatDepth( imageFormat format );

/**
 * Size of an image (in bytes)
 * @ingroup image
 */
inline size_t imageFormatSize( imageFormat format, size_t width, size_t height );

/**
 * @ingroup image
 */
template<typename T> inline imageFormat imageFormatFromType();

template<> inline imageFormat imageFormatFromType<uchar3>();
template<> inline imageFormat imageFormatFromType<uchar4>();
template<> inline imageFormat imageFormatFromType<float3>();
template<> inline imageFormat imageFormatFromType<float4>();


// inline implementations
#include "imageFormat.inl"


#endif

