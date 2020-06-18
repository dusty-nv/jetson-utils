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
	IMAGE_RGBA8,
	IMAGE_RGB32F,
	IMAGE_RGBA32F,

	// BGR
	IMAGE_BGR8,
	IMAGE_BGRA8,
	IMAGE_BGR32F,
	IMAGE_BGRA32F,
	
	// YUV
	IMAGE_YUYV,
	IMAGE_YUY2=IMAGE_YUYV,
	IMAGE_YVYU,
	IMAGE_UYVY,		
	IMAGE_I420,
	IMAGE_YV12,
	IMAGE_NV12,
	
	// Bayer
	IMAGE_BAYER_BGGR,
	IMAGE_BAYER_GBRG,
	IMAGE_BAYER_GRBG,
	IMAGE_BAYER_RGGB,
	
	// grayscale
	IMAGE_GRAY8,	
	IMAGE_GRAY32F,

	// extras
	IMAGE_COUNT,
	IMAGE_UNKNOWN=999,
	IMAGE_DEFAULT=IMAGE_RGBA32F
};

/**
 * @ingroup image
 */
enum imageBaseType
{
	IMAGE_UINT8,
	IMAGE_FLOAT
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
 * @returns true if the imageFormat is a RGB/RGBA format 
 *               (IMAGE_RGB8, IMAGE_RGBA8, IMAGE_RGB32F, IMAGE_RGBA32F)
 *               otherwise, returns false.
 * @ingroup image
 */
inline bool imageFormatIsRGB( imageFormat format );

/**
 * @returns true if the imageFormat is a BGR/BGRA format 
 *               (IMAGE_BGR8, IMAGE_BGRA8, IMAGE_BGR32F, IMAGE_BGRA32F)
 *               otherwise, returns false.
 * @ingroup image
 */
inline bool imageFormatIsRGB( imageFormat format );

/**
 * @returns true if the imageFormat is a YUV format 
 *               (IMAGE_YUYV, IMAGE_YVYU, IMAGE_UYVY, IMAGE_I420, IMAGE_YV12, IMAGE_NV12)
 *               otherwise, returns false.
 * @ingroup image
 */
inline bool imageFormatIsYUV( imageFormat format );

/**
 * @returns true if the imageFormat is grayscale (IMAGE_GRAY8, IMAGE_GRAY32)
 *               otherwise, returns false.
 * @ingroup image
 */
inline bool imageFormatIsGray( imageFormat format );

/**
 * @returns true if the imageFormat is a Bayer format
 *               (IMAGE_BAYER_BGGR, IMAGE_BAYER_GBRG, IMAGE_BAYER_GRBG, IMAGE_BAYER_RGGB)
 *               otherwise, returns false.
 * @ingroup image
 */
inline bool imageFormatIsBayer( imageFormat format );

/**
 * @ingroup image
 */
inline imageBaseType imageFormatBaseType( imageFormat format );

/**
 * @ingroup image
 */
template<typename T> inline imageFormat imageFormatFromType();

template<> inline imageFormat imageFormatFromType<uchar3>();
template<> inline imageFormat imageFormatFromType<uchar4>();
template<> inline imageFormat imageFormatFromType<float3>();
template<> inline imageFormat imageFormatFromType<float4>();

/**
 * @ingroup image
 */
template<imageFormat format> struct imageFormatType;

template<> struct imageFormatType<IMAGE_RGB8>    { typedef uint8_t Base; typedef uchar3 Vector; };
template<> struct imageFormatType<IMAGE_RGBA8>   { typedef uint8_t Base; typedef uchar4 Vector; };

template<> struct imageFormatType<IMAGE_RGB32F>  { typedef float Base; typedef float3 Vector; };
template<> struct imageFormatType<IMAGE_RGBA32F> { typedef float Base; typedef float4 Vector; };


// inline implementations
#include "imageFormat.inl"


#endif

