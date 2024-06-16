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

#include "cudaResize.h"
#include "cudaFilterMode.cuh"


// gpuResize
template<typename T, cudaFilterMode filter>
__global__ void gpuResize( T* input, int inputWidth, int inputHeight, T* output, int outputWidth, int outputHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= outputWidth || y >= outputHeight )
		return;

	output[y * outputWidth + x] = cudaFilterPixel<filter>(input, x, y, inputWidth, inputHeight, outputWidth, outputHeight); 
}

// launchResize
template<typename T>
static cudaError_t launchResize( T* input, size_t inputWidth, size_t inputHeight,
                                 T* output, size_t outputWidth, size_t outputHeight,
                                 cudaFilterMode filter, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	if( outputWidth < inputWidth && outputHeight < inputHeight )
		filter = FILTER_POINT;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	#define launch_resize(filterMode)	\
		gpuResize<T, filterMode><<<gridDim, blockDim, 0, stream>>>(input, inputWidth, inputHeight, output, outputWidth, outputHeight)
	
	if( filter == FILTER_POINT )
		launch_resize(FILTER_POINT);
	else if( filter == FILTER_LINEAR )
		launch_resize(FILTER_LINEAR);

	return CUDA(cudaGetLastError());
}

// cudaResize (uint8 grayscale)
cudaError_t cudaResize( uint8_t* input, size_t inputWidth, size_t inputHeight, uint8_t* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter, cudaStream_t stream )
{
	return launchResize<uint8_t>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter, stream);
}

// cudaResize (float grayscale)
cudaError_t cudaResize( float* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter, cudaStream_t stream )
{
	return launchResize<float>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter, stream);
}

// cudaResize (uchar3)
cudaError_t cudaResize( uchar3* input, size_t inputWidth, size_t inputHeight, uchar3* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter, cudaStream_t stream )
{
	return launchResize<uchar3>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter, stream);
}

// cudaResize (uchar4)
cudaError_t cudaResize( uchar4* input, size_t inputWidth, size_t inputHeight, uchar4* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter, cudaStream_t stream )
{
	return launchResize<uchar4>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter, stream);
}

// cudaResize (float3)
cudaError_t cudaResize( float3* input, size_t inputWidth, size_t inputHeight, float3* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter, cudaStream_t stream )
{
	return launchResize<float3>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter, stream);
}

// cudaResize (float4)
cudaError_t cudaResize( float4* input, size_t inputWidth, size_t inputHeight, float4* output, size_t outputWidth, size_t outputHeight, cudaFilterMode filter, cudaStream_t stream )
{
	return launchResize<float4>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, filter, stream);
}

//-----------------------------------------------------------------------------------
cudaError_t cudaResize( void* input,  size_t inputWidth,  size_t inputHeight,
                        void* output, size_t outputWidth, size_t outputHeight, 
                        imageFormat format, cudaFilterMode filter, cudaStream_t stream )
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
		return cudaResize((uchar3*)input, inputWidth, inputHeight, (uchar3*)output, outputWidth, outputHeight, filter, stream);
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return cudaResize((uchar4*)input, inputWidth, inputHeight, (uchar4*)output, outputWidth, outputHeight, filter, stream);
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return cudaResize((float3*)input, inputWidth, inputHeight, (float3*)output, outputWidth, outputHeight, filter, stream);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return cudaResize((float4*)input, inputWidth, inputHeight, (float4*)output, outputWidth, outputHeight, filter, stream);
	else if( format == IMAGE_GRAY8 )
		return cudaResize((uint8_t*)input, inputWidth, inputHeight, (uint8_t*)output, outputWidth, outputHeight, filter, stream);
	else if( format == IMAGE_GRAY32F )
		return cudaResize((float*)input, inputWidth, inputHeight, (float*)output, outputWidth, outputHeight, filter, stream);

	LogError(LOG_CUDA "cudaResize() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");
	LogError(LOG_CUDA "                    * rgb8, bgr8\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgb32f, bgr32f\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}




