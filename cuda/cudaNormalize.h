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

#ifndef __CUDA_NORMALIZE_H__
#define __CUDA_NORMALIZE_H__


#include "cudaUtility.h"
#include "imageFormat.h"


/**
 * Normalize the pixel intensities of a floating-point grayscale image between two scales.
 * For example, convert an image with values between `[0,1]` to `[0,255]`
 * @param input_range the range of pixel values of the input image (e.g. `[0,1]`)
 * @param output_range the desired range of pixel values of the output image (e.g. `[0,255]`)
 * @ingroup normalization
 */
cudaError_t cudaNormalize( float* input,  const float2& input_range,
					  float* output, const float2& output_range,
					  size_t width,  size_t height );

/**
 * Normalize the pixel intensities of a float3 RGB/BGR image between two scales.
 * For example, convert an image with values between `[0,1]` to `[0,255]`
 * @param input_range the range of pixel values of the input image (e.g. `[0,1]`)
 * @param output_range the desired range of pixel values of the output image (e.g. `[0,255]`)
 * @ingroup normalization
 */
cudaError_t cudaNormalize( float3* input,  const float2& input_range,
					  float3* output, const float2& output_range,
					  size_t  width,  size_t height );

/**
 * Normalize the pixel intensities of a float4 RGBA/BGRA image between two scales.
 * For example, convert an image with values between `[0,1]` to `[0,255]`
 * @param input_range the range of pixel values of the input image (e.g. `[0,1]`)
 * @param output_range the desired range of pixel values of the output image (e.g. `[0,255]`)
 * @ingroup normalization
 */
cudaError_t cudaNormalize( float4* input,  const float2& input_range,
					  float4* output, const float2& output_range,
					  size_t  width,  size_t height );

/**
 * Normalize the pixel intensities of an image between two scales.
 * For example, convert an image with values between `[0,1]` to `[0,255]`
 * @param input_range the range of pixel values of the input image (e.g. `[0,1]`)
 * @param output_range the desired range of pixel values of the output image (e.g. `[0,255]`)
 * @param format the image format - valid formats are gray32f, rgb32f/bgr32f, and rgba32f/bgra32f.
 * @ingroup normalization
 */
cudaError_t cudaNormalize( void* input,  const float2& input_range,
					  void* output, const float2& output_range,
					  size_t width, size_t height, imageFormat format );


#endif

