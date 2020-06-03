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

#include "cudaColorspace.h"

#include "cudaRGB.h"
#include "cudaYUV.h"


// cudaConvertColor
cudaError_t cudaConvertColor( void* input, imageFormat inputFormat,
					     void* output, imageFormat outputFormat,
					     size_t width, size_t height )
{
	if( inputFormat == FORMAT_NV12 )
	{
		if( outputFormat == FORMAT_RGBA8 )
			return CUDA(cudaNV12ToRGBA(input, (uchar4*)output, width, height));
		else if( outputFormat == FORMAT_RGBA32 )
			return CUDA(cudaNV12ToRGBA(input, (float4*)output, width, height));
	}
	else if( inputFormat == FORMAT_RGBA32 )
	{
		if( outputFormat == FORMAT_I420 )
			return CUDA(cudaRGBAToI420((float4*)input, output, width, height));
		else if( outputFormat == FORMAT_YV12 )
			return CUDA(cudaRGBAToYV12((float4*)input, output, width, height));
		else if( outputFormat == FORMAT_RGBA8 )
			return CUDA(cudaRGBA32ToRGBA8((float4*)input, (uchar4*)output, width, height));
		else if( outputFormat == FORMAT_RGB8 )
			return CUDA(cudaRGBA32ToRGB8((float4*)input, (uchar3*)output, width, height));
	}
	else if( inputFormat == FORMAT_RGBA8 )
	{
		if( outputFormat == FORMAT_I420 )
			return CUDA(cudaRGBAToI420((uchar4*)input, output, width, height));
		else if( outputFormat == FORMAT_YV12 )
			return CUDA(cudaRGBAToYV12((uchar4*)input, output, width, height));
	}
	else if( inputFormat == FORMAT_RGB8 )
	{
		if( outputFormat == FORMAT_RGBA8 )
			return CUDA(cudaRGB8ToRGBA32((uchar3*)input, (float4*)output, width, height));
	}
	else if( inputFormat == FORMAT_YUYV )
	{
		if( outputFormat == FORMAT_RGBA8 )
			return CUDA(cudaYUYVToRGBA(input, (uchar4*)output, width, height));
		else if( outputFormat == FORMAT_RGBA32 )
			return CUDA(cudaYUYVToRGBA(input, (float4*)output, width, height));
	}
	else if( inputFormat == FORMAT_UYVY )
	{
		if( outputFormat == FORMAT_RGBA8 )
			return CUDA(cudaUYVYToRGBA(input, (uchar4*)output, width, height));
		else if( outputFormat == FORMAT_RGBA32 )
			return CUDA(cudaUYVYToRGBA(input, (float4*)output, width, height));
	}

	printf(LOG_CUDA "cudaColorConvert() -- invalid input/output format combination (%s->%s)\n", imageFormatToStr(inputFormat), imageFormatToStr(inputFormat));
	return cudaErrorInvalidValue;
}


