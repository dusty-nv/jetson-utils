/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __CUDA_FILTER_MODE_CUH__
#define __CUDA_FILTER_MODE_CUH__


#include "cudaFilterMode.h"
#include "cudaMath.h"


//////////////////////////////////////////////////////////////////////////////////////////
/// @name CUDA device functions for reading a single pixel, either in HWC or CHW layout.
/// @ingroup cudaFilter
//////////////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * CUDA device function for reading a pixel from an image, either in HWC or CHW layout.
 *
 * @param input pointer to image in CUDA device memory
 * @param x desired x-coordinate to sample
 * @param y desired y-coordinate to sample
 * @param width width of the input image
 * @param height height of the input image
 *
 * @returns the raw pixel data from the input image
 */
template<cudaDataFormat layout, typename T>
__device__ inline T cudaReadPixel( T* input, int x, int y, int width, int height )
{
	return input[y * width + x];
}

template<> __device__ inline 
float2 cudaReadPixel<FORMAT_CHW>( float2* input, int x, int y, int width, int height )
{
	float* ptr = (float*)input;
	const int offset = y * width + x;
	return make_float2(ptr[offset], ptr[width * height + offset]);
}

template<> __device__ inline 
float3 cudaReadPixel<FORMAT_CHW>( float3* input, int x, int y, int width, int height )
{
	float* ptr = (float*)input;
	const int offset = y * width + x;
	const int pixels = width * height;
	return make_float3(ptr[offset], ptr[pixels + offset], ptr[pixels * 2 + offset]);
}

template<> __device__ inline 
float4 cudaReadPixel<FORMAT_CHW>( float4* input, int x, int y, int width, int height )
{
	float* ptr = (float*)input;
	const int offset = y * width + x;
	const int pixels = width * height;
	return make_float4(ptr[offset], ptr[pixels + offset], ptr[pixels * 2 + offset], ptr[pixels * 3 + offset]);
}

///@}

/**
 * CUDA device function for sampling a pixel with bilinear or point filtering.
 * cudaFilterPixel() is for use inside of other CUDA kernels, and accepts a
 * cudaFilterMode template parameter which sets the filtering mode, in addition
 * to a cudaDataFormat template parameter which sets the format (HWC or CHW).
 *
 * @param input pointer to image in CUDA device memory
 * @param x desired x-coordinate to sample
 * @param y desired y-coordinate to sample
 * @param width width of the input image
 * @param height height of the input image
 *
 * @returns the filtered pixel from the input image
 * @ingroup cudaFilter
 */ 
template<cudaFilterMode filter, cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel( T* input, float x, float y, int width, int height )
{
	if( filter == FILTER_POINT )
	{
		const int x1 = int(x);
		const int y1 = int(y);

		return cudaReadPixel<format>(input, x1, y1, width, height); //input[y1 * width + x1];
	}
	else // FILTER_LINEAR
	{
		const float bx = x - 0.5f;
		const float by = y - 0.5f;

		const float cx = bx < 0.0f ? 0.0f : bx;
		const float cy = by < 0.0f ? 0.0f : by;

		const int x1 = int(cx);
		const int y1 = int(cy);
			
		const int x2 = x1 >= width - 1 ? x1 : x1 + 1;	// bounds check
		const int y2 = y1 >= height - 1 ? y1 : y1 + 1;
		
		const T samples[4] = {
			cudaReadPixel<format>(input, x1, y1, width, height),   //input[y1 * width + x1],
			cudaReadPixel<format>(input, x2, y1, width, height),   //input[y1 * width + x2],
			cudaReadPixel<format>(input, x1, y2, width, height),   //input[y2 * width + x1],
			cudaReadPixel<format>(input, x2, y2, width, height) }; //input[y2 * width + x2] };

		// compute bilinear weights
		const float x1d = cx - float(x1);
		const float y1d = cy - float(y1);

		const float x1f = 1.0f - x1d;
		const float y1f = 1.0f - y1d;

		const float x2f = 1.0f - x1f;
		const float y2f = 1.0f - y1f;

		const float x1y1f = x1f * y1f;
		const float x1y2f = x1f * y2f;
		const float x2y1f = x2f * y1f;
		const float x2y2f = x2f * y2f;

		return samples[0] * x1y1f + samples[1] * x2y1f + samples[2] * x1y2f + samples[3] * x2y2f;
	}
}

/**
 * CUDA device function for sampling a pixel with bilinear or point filtering.
 * cudaFilterPixel() is for use inside of other CUDA kernels, and samples a
 * pixel from an input image from the scaled coordinates of an output image.
 *
 * @param input pointer to image in CUDA device memory
 * @param x desired x-coordinate to sample (in coordinate space of output image)
 * @param y desired y-coordinate to sample (in coordinate space of output image)
 * @param input_width width of the input image
 * @param input_height height of the input image
 * @param output_width width of the output image
 * @param output_height height of the output image
 *
 * @returns the filtered pixel from the input image
 * @ingroup cudaFilter
 */ 
template<cudaFilterMode filter, cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel( T* input, int x, int y,
						       int input_width, int input_height,
						       int output_width, int output_height )
{
	const float px = float(x) / float(output_width) * float(input_width);
	const float py = float(y) / float(output_height) * float(input_height);

	return cudaFilterPixel<filter, format>(input, px, py, input_width, input_height);
}


#endif


