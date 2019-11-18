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

#include "cudaPointCloud.h"
#include "cudaMath.h"


// gpuPointCloudExtract
template<bool useRGB>
__global__ void gpuPointCloudExtract( float3* points, float* depth, float4* rgba, 
							   int width, int height, float2 fx, float2 cx )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;

	if( x >= width || y >= height )
		return;

	// read depth map sample
	const float depth_sample = depth[i];

	// determine output address
	float3* output = points + i * (useRGB + 1);

	// apply depth calibration
	output[0] = make_float3((float(x) - cx.x) * depth_sample / fx.x,
				    	    (float(y) - cx.y) * depth_sample / fx.y * -1.0f,
				     	depth_sample * -1.0f);

	// output RGB if needed
	if( useRGB )
		output[1] = make_float3(rgba[i]);
}


// Extract
bool cudaPointCloud::Extract( float* depth, float4* rgba, uint32_t width, uint32_t height,
					 	const float2& focalLength, const float2& principalPoint )
{
	if( !depth )
	{
		printf(LOG_CUDA "cudaPointCloud::Extract() -- depth map is NULL\n");
		return false;
	}

	if( width == 0 || height == 0 )
	{
		printf(LOG_CUDA "cudaPointCloud::Extract() -- width/height parameters are zero\n");
		return false;
	}

	// determine if RGB used
	if( rgba != NULL )
		mHasRGB = true;

	// allocate memory
	const uint32_t numPoints = width * height;

	if( !Reserve(numPoints) )
		return false;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	if( mHasRGB )
		gpuPointCloudExtract<true><<<gridDim, blockDim>>>(mPointsGPU, depth, rgba, width, height, focalLength, principalPoint);
	else
		gpuPointCloudExtract<false><<<gridDim, blockDim>>>(mPointsGPU, depth, rgba, width, height, focalLength, principalPoint);

	// check for launch errors
	if( CUDA_FAILED(cudaGetLastError()) )
	{
		printf(LOG_CUDA "cudaPointCloud::Extract() -- failed to extra point cloud with CUDA\n");
		return false;
	}
	
	return true;
}







