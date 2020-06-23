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

#ifndef __CUDA_RESIZE_H__
#define __CUDA_RESIZE_H__


#include "cudaUtility.h"
#include "imageFormat.h"


/**
 * Rescale a uint8 grayscale image on the GPU.
 * @ingroup resize
 */
cudaError_t cudaResize( uint8_t* input,  size_t inputWidth,  size_t inputHeight,
				    uint8_t* output, size_t outputWidth, size_t outputHeight );

/**
 * Rescale a floating-point grayscale image on the GPU.
 * @ingroup resize
 */
cudaError_t cudaResize( float* input,  size_t inputWidth,  size_t inputHeight,
				    float* output, size_t outputWidth, size_t outputHeight );

/**
 * Rescale a uchar3 RGB/BGR image on the GPU.
 * @ingroup resize
 */
cudaError_t cudaResize( uchar3* input,  size_t inputWidth,  size_t inputHeight,
				    uchar3* output, size_t outputWidth, size_t outputHeight );

/**
 * Rescale a float3 RGB/BGR image on the GPU.
 * @ingroup resize
 */
cudaError_t cudaResize( float3* input,  size_t inputWidth,  size_t inputHeight,
				    float3* output, size_t outputWidth, size_t outputHeight );

/**
 * Rescale a uchar4 RGBA/BGRA image on the GPU.
 * @ingroup resize
 */
cudaError_t cudaResize( uchar4* input,  size_t inputWidth,  size_t inputHeight,
				    uchar4* output, size_t outputWidth, size_t outputHeight );

/**
 * Rescale a float4 RGBA/BGRA image on the GPU.
 * @ingroup resize
 */
cudaError_t cudaResize( float4* input,  size_t inputWidth,  size_t inputHeight,
				    float4* output, size_t outputWidth, size_t outputHeight );

/**
 * Rescale an image on the GPU (supports grayscale, RGB/BGR, RGBA/BGRA)
 * @ingroup resize
 */
cudaError_t cudaResize( void* input,  size_t inputWidth,  size_t inputHeight,
				    void* output, size_t outputWidth, size_t outputHeight, 
				    imageFormat format );

#endif

