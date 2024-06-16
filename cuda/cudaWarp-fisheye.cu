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

#include "cudaWarp.h"


// cudaFisheye
template<typename T>
__global__ void cudaFisheye( T* input, T* output, int width, int height, float focus )
{
	const int2 uv_out = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
				               blockDim.y * blockIdx.y + threadIdx.y);
						   
	if( uv_out.x >= width || uv_out.y >= height )
		return;
	
	const float fWidth  = width;
	const float fHeight = height;
	
	// convert to cartesian coordinates
	const float cx = ((uv_out.x / fWidth) - 0.5f)  * 2.0f;	
	const float cy = (0.5f - (uv_out.y / fHeight)) * 2.0f;

	const float theta = atan2f(cy, cx);
	const float r     = atanf(sqrtf(cx*cx+cy*cy) * focus);
	
	const float tx = r * __cosf(theta);
	const float ty = r * __sinf(theta);
	
	// convert back out of cartesian coordinates
	float u = (tx * 0.5f + 0.5f) * fWidth;
	float v = (0.5f - (ty * 0.5f)) * fHeight;

	if( u < 0.0f ) u = 0.0f;
	if( v < 0.0f ) v = 0.0f;

	if( u > fWidth  - 1.0f ) u = fWidth - 1.0f;
	if( v > fHeight - 1.0f ) v = fHeight - 1.0f;
	
	output[uv_out.y * width + uv_out.x] = input[(int)v * width + (int)u];
} 


// cudaWarpFisheye
cudaError_t cudaWarpFisheye( uchar4* input, uchar4* output, uint32_t width, uint32_t height, float focus, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	cudaFisheye<<<gridDim, blockDim, 0, stream>>>(input, output, width, height, focus);

	return CUDA(cudaGetLastError());
}


// cudaWarpFisheye
cudaError_t cudaWarpFisheye( float4* input, float4* output, uint32_t width, uint32_t height, float focus, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	cudaFisheye<<<gridDim, blockDim, 0, stream>>>(input, output, width, height, focus);

	return CUDA(cudaGetLastError());
}


