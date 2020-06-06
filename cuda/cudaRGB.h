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
/// @name 8-bit RGB/BGR to 8-bit RGBA/BGRA (or vice-versa)
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert uchar3 RGB/BGR image to uchar4 RGBA/BGRA image
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGB8ToRGBA8( uchar3* input, uchar4* output, size_t width, size_t height, bool swapRedBlue=false );

/**
 * Convert uchar4 RGBA/BGRA image to uchar3 RGB/BGR image
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGBA8ToRGB8( uchar4* input, uchar3* output, size_t width, size_t height, bool swapRedBlue=false );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name Floating-point RGB/BGR to floating-point RGBA/BGRA (or vice versa)
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert float3 RGB/BGR image into float4 RGBA/BGRA image.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGB32ToRGBA32( float3* input, float4* output, size_t width, size_t height, bool swapRedBlue=false );

/**
 * Convert float4 RGBA/BGRA image into float3 RGB/BGR image.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToRGB32( float4* input, float3* output, size_t width, size_t height, bool swapRedBlue=false );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name 8-bit images to floating-point images
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert uchar3 RGB/BGR image to float3 RGB/BGR image
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGB8ToRGB32( uchar3* input, float3* output, size_t width, size_t height, bool swapRedBlue=false );

/**
 * Convert uchar3 RGB/BGR image to float4 RGBA/BGRA image
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGB8ToRGBA32( uchar3* input, float4* output, size_t width, size_t height, bool swapRedBlue=false );

/**
 * Convert uchar4 RGBA/BGRA image to float3 RGB/BGR image
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGBA8ToRGB32( uchar4* input, float3* output, size_t width, size_t height, bool swapRedBlue=false );

/**
 * Convert uchar4 RGBA/BGRA image to float4 RGBA/BGRA image
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGBA8ToRGBA32( uchar4* input, float4* output, size_t width, size_t height, bool swapRedBlue=false );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name Floating-point images to 8-bit images
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert float3 RGB/BGR image into uchar3 RGB/BGR image.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @param pixelRange specifies the floating-point pixel value range of the input image, 
 *                   which is used to rescale the fixed-point pixel outputs to [0-255].
 *                   The default input range is [0,255], where no rescaling occurs.
 *                   Other common input ranges are [-1, 1] or [0,1].
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGB32ToRGB8( float3* input, uchar3* output, size_t width, size_t height, 
					    bool swapRedBlue=false, const float2& pixelRange=make_float2(0,255) );

/**
 * Convert float3 RGB/BGR image into uchar4 RGBA/BGRA image.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @param pixelRange specifies the floating-point pixel value range of the input image, 
 *                   which is used to rescale the fixed-point pixel outputs to [0-255].
 *                   The default input range is [0,255], where no rescaling occurs.
 *                   Other common input ranges are [-1, 1] or [0,1].
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGB32ToRGBA8( float3* input, uchar4* output, size_t width, size_t height, 
						bool swapRedBlue=false, const float2& pixelRange=make_float2(0,255) );

/**
 * Convert float4 RGBA/BGRA image into uchar3 image.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @param pixelRange specifies the floating-point pixel value range of the input image, 
 *                   which is used to rescale the fixed-point pixel outputs to [0-255].
 *                   The default input range is [0,255], where no rescaling occurs.
 *                   Other common input ranges are [-1, 1] or [0,1].
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToRGB8( float4* input, uchar3* output, size_t width, size_t height, 
						bool swapRedBlue=false, const float2& pixelRange=make_float2(0,255) );

/**
 * Convert float4 RGBA/BGRA image into uchar4 RGBA/BGRA image.
 *
 * @param swapRedBlue if true, swap the input's red and blue channels in the output -
 *                    i.e if the input is RGB and output is BGR, or vice versa.  
 *                    The default is false, and the channels will remain the same.
 *
 * @param pixelRange specifies the floating-point pixel value range of the input image, 
 *                   which is used to rescale the fixed-point pixel outputs to [0-255].
 *                   The default input range is [0,255], where no rescaling occurs.
 *                   Other common input ranges are [-1, 1] or [0,1].
 *
 * @ingroup colorspace
 */
cudaError_t cudaRGBA32ToRGBA8( float4* input, uchar4* output, size_t width, size_t height, 
						 bool swapRedBlue=false, const float2& pixelRange=make_float2(0,255) );

///@}


#endif

