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


// gpuIntrinsicWarp
template<typename T>
__global__ void gpuIntrinsicWarp( T* input, T* output, int width, int height,
						    float2 focalLength, float2 principalPoint, float k1, float k2, float p1, float p2)
{
	const int2 uv_out = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
				               blockDim.y * blockIdx.y + threadIdx.y);
						   
	if( uv_out.x >= width || uv_out.y >= height )
		return;
	
	const float u = uv_out.x;
	const float v = uv_out.y;

	const float _fx = 1.0f / focalLength.x;
	const float _fy = 1.0f / focalLength.y;
	
	const float y      = (v - principalPoint.y)*_fy;
	const float y2     = y*y;
	const float _2p1y  = 2.0*p1*y;
	const float _3p1y2 = 3.0*p1*y2;
	const float p2y2   = p2*y2;

	const float x  = (u - principalPoint.x)*_fx;
	const float x2 = x*x;
	const float r2 = x2 + y2;
	const float d  = 1.0 + (k1 + k2*r2)*r2;
	const float _u = focalLength.x*(x*(d + _2p1y) + p2y2 + (3.0*p2)*x2) + principalPoint.x;
	const float _v = focalLength.y*(y*(d + (2.0*p2)*x) + _3p1y2 + p1*x2) + principalPoint.y;

	const int2 uv_in = make_int2( _u, _v );
	
	if( uv_in.x >= width || uv_in.y >= height || uv_in.x < 0 || uv_in.y < 0 )
		return;

	output[uv_out.y * width + uv_out.x] = input[uv_in.y * width + uv_in.x];
} 


// cudaWarpIntrinsic
cudaError_t cudaWarpIntrinsic( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
						 const float2& focalLength, const float2& principalPoint, const float4& distortion )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuIntrinsicWarp<<<gridDim, blockDim>>>(input, output, width, height,
									focalLength, principalPoint,
									distortion.x, distortion.y, distortion.z, distortion.w);

	return CUDA(cudaGetLastError());
}


// cudaWarpIntrinsic
cudaError_t cudaWarpIntrinsic( float4* input, float4* output, uint32_t width, uint32_t height,
						 const float2& focalLength, const float2& principalPoint, const float4& distortion )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuIntrinsicWarp<<<gridDim, blockDim>>>(input, output, width, height,
									focalLength, principalPoint,
									distortion.x, distortion.y, distortion.z, distortion.w);

	return CUDA(cudaGetLastError());
}






