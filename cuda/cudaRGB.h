/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __CUDA_RGB_CONVERT_H
#define __CUDA_RGB_CONVERT_H


#include "cudaUtility.h"



//////////////////////////////////////////////////////////////////////////////////
/// @name 8-bit RGB/BGR to Floating-point RGBA
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{
	
/**
 * Convert 8-bit fixed-point RGB image to 32-bit floating-point RGBA image
 * @ingroup colorspace
 */
cudaError_t cudaRGB8ToRGBA32( uchar3* input, float4* output, size_t width, size_t height );

/**
 * Convert 8-bit fixed-point BGR image to 32-bit floating-point RGBA image
 * @ingroup colorspace
 */
cudaError_t cudaBGR8ToRGBA32( uchar3* input, float4* output, size_t width, size_t height );

///@}

//////////////////////////////////////////////////////////////////////////////////
/// @name Floating-point RGBA to 8-bit RGB/RGBA
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert 32-bit floating-point RGBA image into 8-bit fixed-point RGB image.
 * Assumes 0.0-255.0f input range, output range is 0-255.
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToRGB8( float4* input, uchar3* output, size_t width, size_t height );

/**
 * Convert 32-bit floating-point RGBA image into 8-bit fixed-point RGB image,
 * with the floating-point input range specified by the user.  Output range is 0-255.
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToRGB8( float4* input, uchar3* output, size_t width, size_t height, const float2& inputRange );

/**
 * Convert 32-bit floating-point RGBA image into 8-bit fixed-point RGBA image.
 * Assumes 0.0-255.0f input range, output range is 0-255.
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToRGBA8( float4* input, uchar4* output, size_t width, size_t height );

/**
 * Convert 32-bit floating-point RGBA image into 8-bit fixed-point RGBA image,
 * with the floating-point input range specified by the user.  Output range is 0-255.
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToRGBA8( float4* input, uchar4* output, size_t width, size_t height, const float2& inputRange );

///@}

//////////////////////////////////////////////////////////////////////////////////
/// @name Floating-point RGBA to 8-bit BGR/BGRA
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{
	
/**
 * Convert 32-bit floating-point RGBA image into 8-bit fixed-point BGR image.
 * Assumes 0.0-255.0f input range, output range is 0-255.
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToBGR8( float4* input, uchar3* output, size_t width, size_t height );

/**
 * Convert 32-bit floating-point RGBA image into 8-bit fixed-point BGR image,
 * with the floating-point input range specified by the user.  Output range is 0-255.
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToBGR8( float4* input, uchar3* output, size_t width, size_t height, const float2& inputRange );

/**
 * Convert 32-bit floating-point RGBA image into 8-bit fixed-point BGRA image.
 * Assumes 0.0-255.0f input range, output range is 0-255.
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToBGRA8( float4* input, uchar4* output, size_t width, size_t height );

/**
 * Convert 32-bit floating-point RGBA image into 8-bit fixed-point BGRA image,
 * with the floating-point input range specified by the user.  Output range is 0-255.
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToBGRA8( float4* input, uchar4* output, size_t width, size_t height, const float2& inputRange );

///@}

#endif

