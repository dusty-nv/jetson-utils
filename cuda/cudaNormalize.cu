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

#include "cudaNormalize.h"
#include "cudaVector.h"


// gpuNormalize
template <typename T>
__global__ void gpuNormalize( T* input, T* output, int width, int height, 
					     float2 input_range, float scaling_factor )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T px = input[ y * width + x ];

	#define rescale(v) ((v - input_range.x) * scaling_factor)

	output[y*width+x] = make_vec<T>(rescale(px.x),
							  rescale(px.y),
							  rescale(px.z),
							  rescale(alpha(px, input_range.y)));
}

template<typename T>
static cudaError_t launchNormalizeRGB( T* input, const float2& input_range,
						  T* output, const float2& output_range,
						  size_t  width,  size_t height )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0  )
		return cudaErrorInvalidValue;

	const float multiplier = output_range.y / input_range.y;

	// launch kernel
	const dim3 blockDim(32,8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuNormalize<T><<<gridDim, blockDim>>>(input, output, width, height, input_range, multiplier);

	return CUDA(cudaGetLastError());
}


// cudaNormalize (float3)
cudaError_t cudaNormalize( float3* input, const float2& input_range,
					  float3* output, const float2& output_range,
					  size_t  width,  size_t height )
{
	return launchNormalizeRGB<float3>(input, input_range, output, output_range, width, height);
}


// cudaNormalize (float4)
cudaError_t cudaNormalize( float4* input, const float2& input_range,
					  float4* output, const float2& output_range,
					  size_t  width,  size_t height )
{
	return launchNormalizeRGB<float4>(input, input_range, output, output_range, width, height);
}


//-----------------------------------------------------------------------------------
template <typename T>
__global__ void gpuNormalizeGray( T* input, T* output, int width, int height, 
					     float2 input_range, float scaling_factor )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T px = rescale(input[ y * width + x ]);
	output[y*width+x] = px;
}

template<typename T>
static cudaError_t launchNormalizeGray( T* input, const float2& input_range,
						  	     T* output, const float2& output_range,
						  		size_t width, size_t height )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0  )
		return cudaErrorInvalidValue;

	const float multiplier = output_range.y / input_range.y;

	// launch kernel
	const dim3 blockDim(32,8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuNormalizeGray<T><<<gridDim, blockDim>>>(input, output, width, height, input_range, multiplier);

	return CUDA(cudaGetLastError());
}


// cudaNormalize (float)
cudaError_t cudaNormalize( float* input, const float2& input_range,
					  float* output, const float2& output_range,
					  size_t width, size_t height )
{
	return launchNormalizeGray<float>(input, input_range, output, output_range, width, height);
}


//-----------------------------------------------------------------------------------
cudaError_t cudaNormalize( void* input,  const float2& input_range,
					  void* output, const float2& output_range,
					  size_t width, size_t height, imageFormat format )
{
	if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return cudaNormalize((float3*)input, input_range, (float3*)output, output_range, width, height);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return cudaNormalize((float4*)input, input_range, (float4*)output, output_range, width, height);
	else if( format == IMAGE_GRAY32F )
		return cudaNormalize((float*)input, input_range, (float*)output, output_range, width, height);

	LogError(LOG_CUDA "cudaNormalize() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                   supported formats are:\n");
	LogError(LOG_CUDA "                       * gray32f\n");
	LogError(LOG_CUDA "                       * rgb32f, bgr32f\n");
	LogError(LOG_CUDA "                       * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}


