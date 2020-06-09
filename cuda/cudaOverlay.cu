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

#include "cudaOverlay.h"


template<typename T>
__global__ void gpuRectFill( T* input, T* output, int width, int height,
					    float4* rects, int numRects, float4 color ) 
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	T px = input[ y * width + x ];

	const float fx = x;
	const float fy = y;
	
	const float alpha = color.w / 255.0f;
	const float ialph = 1.0f - alpha;
	
	for( int nr=0; nr < numRects; nr++ )
	{
		const float4 r = rects[nr];
	
		if( fy >= r.y && fy <= r.w && fx >= r.x && fx <= r.z )
		{
			px.x = alpha * color.x + ialph * px.x;
			px.y = alpha * color.y + ialph * px.y;
			px.z = alpha * color.z + ialph * px.z;
		}
	}
	
	output[y * width + x] = px;	 
}

template<typename T>
__global__ void gpuRectFillBox( T* input, T* output, int imgWidth, int imgHeight, int x0, int y0, int boxWidth, int boxHeight, const float4 color ) 
{
	const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	if( box_x >= boxWidth || box_y >= boxHeight )
		return;

	const int x = box_x + x0;
	const int y = box_y + y0;

	if( x >= imgWidth || y >= imgHeight )
		return;

	T px = input[ y * imgWidth + x ];

	const float alpha = color.w / 255.0f;
	const float ialph = 1.0f - alpha;

	px.x = alpha * color.x + ialph * px.x;
	px.y = alpha * color.y + ialph * px.y;
	px.z = alpha * color.z + ialph * px.z;
	
	output[y * imgWidth + x] = px;
}

template<typename T>
cudaError_t launchRectFill( T* input, T* output, uint32_t width, uint32_t height, float4* rects, int numRects, const float4& color )
{
	if( !input || !output || width == 0 || height == 0 || !rects || numRects == 0 )
		return cudaErrorInvalidValue;

	// if input and output are the same image, then we can use the faster method
	// which draws 1 box per kernel, but doesn't copy pixels that aren't inside boxes
	if( input == output )
	{
		for( int n=0; n < numRects; n++ )
		{
			const int boxWidth = (int)(rects[n].z - rects[n].x);
			const int boxHeight = (int)(rects[n].w - rects[n].y);

			// launch kernel
			const dim3 blockDim(8, 8);
			const dim3 gridDim(iDivUp(boxWidth,blockDim.x), iDivUp(boxHeight,blockDim.y));

			gpuRectFillBox<T><<<gridDim, blockDim>>>(input, output, width, height, (int)rects[n].x, (int)rects[n].y, boxWidth, boxHeight, color); 
		}
	}
	else
	{
		// launch kernel
		const dim3 blockDim(8, 8);
		const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

		gpuRectFill<T><<<gridDim, blockDim>>>(input, output, width, height, rects, numRects, color);
	}

	return cudaGetLastError();
}

// cudaRectFill
cudaError_t cudaRectFill( void* input, void* output, imageFormat format, uint32_t width, uint32_t height, float4* rects, int numRects, const float4& color )
{
	if( !input || !output || width == 0 || height == 0 || !rects || numRects == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	if( format == IMAGE_RGB8 )
		return launchRectFill<uchar3>((uchar3*)input, (uchar3*)output, width, height, rects, numRects, color); 
	else if( format == IMAGE_RGBA8 )
		return launchRectFill<uchar4>((uchar4*)input, (uchar4*)output, width, height, rects, numRects, color); 
	else if( format == IMAGE_RGB32F )
		return launchRectFill<float3>((float3*)input, (float3*)output, width, height, rects, numRects, color); 
	else if( format == IMAGE_RGBA32F )
		return launchRectFill<float4>((float4*)input, (float4*)output, width, height, rects, numRects, color); 
	else
		return cudaErrorInvalidValue;
}
