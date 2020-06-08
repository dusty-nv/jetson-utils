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

#ifndef __CUDA_COLORSPACE_H__
#define __CUDA_COLORSPACE_H__

#include "cudaUtility.h"
#include "imageFormat.h"


/**
 * @ingroup colorspace
 */
cudaError_t cudaConvertColor( void* input, imageFormat inputFormat,
					     void* output, imageFormat outputFormat,
					     size_t width, size_t height,
						 const float2& pixel_range=make_float2(0,255));

/**
 * @ingroup colorspace
 */
template<typename T_in, typename T_out> 
cudaError_t cudaConvertColor( T_in* input, T_out* output,
					     size_t width, size_t height,
						const float2& pixel_range=make_float2(0,255))	
{ 
	return cudaConvertColor(input, imageFormatFromType<T_in>(), output, imageFormatFromType<T_out>(), width, height, pixel_range); 
}
	

#endif

