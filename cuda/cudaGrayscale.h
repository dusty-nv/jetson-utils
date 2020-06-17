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

#ifndef __CUDA_GRAYSCALE_CONVERT_H
#define __CUDA_GRAYSCALE_CONVERT_H


#include "cudaUtility.h"


//////////////////////////////////////////////////////////////////////////////////
/// @name 8-bit grayscale to floating-point grayscale (and vice versa)
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert uint8 grayscale image into float grayscale.
 * @ingroup colorspace
 */
cudaError_t cudaGray8ToGray32( uint8_t* input, float* output, size_t width, size_t height );

/**
 * Convert float grayscale image into uint8 grayscale.
 *
 * @param pixelRange specifies the floating-point pixel value range of the input image, 
 *                   which is used to rescale the fixed-point pixel outputs to [0,255].
 *                   The default input range is [0,255], where no rescaling occurs.
 *                   Other common input ranges are [-1, 1] or [0,1].
 *
 * @ingroup colorspace
 */
cudaError_t cudaGray32ToGray8( float* input, uint8_t* output, size_t width, size_t height, 
						 const float2& pixelRange=make_float2(0,255) );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name RGB/BGR to 8-bit grayscale
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert uchar3 RGB/BGR image into uint8 grayscale.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels. 
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGB8ToGray8( uchar3* input, uint8_t* output, size_t width, size_t height, bool swapRedBlue=false );

/**
 * Convert uchar4 RGBA/BGRA image into uint8 grayscale.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels. 
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGBA8ToGray8( uchar4* input, uint8_t* output, size_t width, size_t height, bool swapRedBlue=false );

/**
 * Convert float3 RGB/BGR image into uint8 grayscale.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels. 
 *                    The default is false, and the channels will remain the same.
 *
 * @param pixelRange specifies the floating-point pixel value range of the input image, 
 *                   which is used to rescale the fixed-point pixel outputs to [0,255].
 *                   The default input range is [0,255], where no rescaling occurs.
 *                   Other common input ranges are [-1, 1] or [0,1].
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGB32ToGray8( float3* input, uint8_t* output, size_t width, size_t height, 
							  bool swapRedBlue=false, const float2& pixelRange=make_float2(0,255) );
						
/**
 * Convert float4 RGBA/BGRA image into uint8 grayscale.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels. 
 *                    The default is false, and the channels will remain the same.
 *
 * @param pixelRange specifies the floating-point pixel value range of the input image, 
 *                   which is used to rescale the fixed-point pixel outputs to [0,255].
 *                   The default input range is [0,255], where no rescaling occurs.
 *                   Other common input ranges are [-1, 1] or [0,1].
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToGray8( float4* input, uint8_t* output, size_t width, size_t height, 
							   bool swapRedBlue=false, const float2& pixelRange=make_float2(0,255) );
							   
///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name RGB/BGR to floating-point grayscale
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert uchar3 RGB/BGR image into float grayscale.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels. 
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGB8ToGray32( uchar3* input, float* output, size_t width, size_t height, bool swapRedBlue=false );

/**
 * Convert uchar4 RGBA/BGRA image into float grayscale.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels. 
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGBA8ToGray32( uchar4* input, float* output, size_t width, size_t height, bool swapRedBlue=false );

/**
 * Convert float3 RGB/BGR image into float grayscale.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels. 
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGB32ToGray32( float3* input, float* output, size_t width, size_t height, bool swapRedBlue=false );

/**
 * Convert float4 RGB/BGR image into float grayscale.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels. 
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToGray32( float4* input, float* output, size_t width, size_t height, bool swapRedBlue=false );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name 8-bit grayscale to RGB/BGR
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert uint8 grayscale image into uchar3 RGB/BGR.
 * @ingroup colorspace
 */
cudaError_t cudaGray8ToRGB8( uint8_t* input, uchar3* output, size_t width, size_t height );

/**
 * Convert uint8 grayscale image into uchar4 RGB/BGR.
 * @ingroup colorspace
 */
cudaError_t cudaGray8ToRGBA8( uint8_t* input, uchar4* output, size_t width, size_t height );

/**
 * Convert uint8 grayscale image into float3 RGB/BGR.
 * @ingroup colorspace
 */
cudaError_t cudaGray8ToRGB32( uint8_t* input, float3* output, size_t width, size_t height );

/**
 * Convert uint8 grayscale image into float4 RGB/BGR.
 * @ingroup colorspace
 */
cudaError_t cudaGray8ToRGBA32( uint8_t* input, float4* output, size_t width, size_t height );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name Floating-point grayscale to RGB/BGR
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert float grayscale image into uchar3 RGB/BGR.
 *
 * @param pixelRange specifies the floating-point pixel value range of the input image, 
 * @param pixelRange specifies the floating-point pixel value range of the input image, 
 *                   which is used to rescale the fixed-point pixel outputs to [0,255].
 *                   The default input range is [0,255], where no rescaling occurs.
 *                   Other common input ranges are [-1, 1] or [0,1].
 *
 * @ingroup colorspace
 */
cudaError_t cudaGray32ToRGB8( float* input, uchar3* output, size_t width, size_t height, 
						const float2& pixelRange=make_float2(0,255) );

/**
 * Convert float grayscale image into uchar4 RGB/BGR.
 *
 * @param pixelRange specifies the floating-point pixel value range of the input image, 
 *                   which is used to rescale the fixed-point pixel outputs to [0,255].
 *                   The default input range is [0,255], where no rescaling occurs.
 *                   Other common input ranges are [-1, 1] or [0,1].
 *
 * @ingroup colorspace
 */
cudaError_t cudaGray32ToRGBA8( float* input, uchar4* output, size_t width, size_t height, 
						 const float2& pixelRange=make_float2(0,255) );

/**
 * Convert float grayscale image into float3 RGB/BGR.
 * @ingroup colorspace
 */
cudaError_t cudaGray32ToRGB32( float* input, float3* output, size_t width, size_t height );

/**
 * Convert float grayscale image into float4 RGB/BGR.
 * @ingroup colorspace
 */
cudaError_t cudaGray32ToRGBA32( float* input, float4* output, size_t width, size_t height );

///@}

#endif
