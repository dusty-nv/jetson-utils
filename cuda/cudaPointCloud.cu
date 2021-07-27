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
#include "cudaColormap.h"

#include "logging.h"


// float4 to uchar3
inline __host__ __device__ uchar3 make_uchar3( float4 a )
{
    return make_uchar3(fminf(a.x, 255.0f), fminf(a.y, 255.0f), fminf(a.z, 255.0f));
}


// gpuPointCloudExtract
template<bool useRGB>
__global__ void gpuPointCloudExtract( float* depth, float4* rgba, int width, int height, 
							   float2 fx, float2 cx, cudaPointCloud::Vertex* points )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;

	if( x >= width || y >= height )
		return;

	// read depth map sample
	const float depth_sample = depth[i];

	// create output point
	cudaPointCloud::Vertex point;

	point.classID = 0;	// not applied yet

	// apply depth calibration
	point.pos = make_float3((float(x) - cx.x) * depth_sample / fx.x,
				    	    (float(y) - cx.y) * depth_sample / fx.y * -1.0f,
				     	depth_sample * -1.0f);

	// read RGB if needed
	if( useRGB )
		point.color = make_uchar3(rgba[i]);
	else
		point.color = make_uchar3(255,255,255);

	// save the point
	points[i] = point;
}


// Extract
bool cudaPointCloud::Extract( float* depth, uint32_t depth_width, uint32_t depth_height,
						float4* rgba, uint32_t color_width, uint32_t color_height )
{
	if( !depth )
	{
		LogError(LOG_CUDA "cudaPointCloud::Extract() -- depth map is NULL\n");
		return false;
	}

	if( depth_width == 0 || depth_height == 0 || color_width == 0 || color_height == 0 )
	{
		LogError(LOG_CUDA "cudaPointCloud::Extract() -- depth width/height parameters are zero\n");
		return false;
	}

	if( rgba != NULL && (color_width == 0 || color_height == 0) )
	{
		LogError(LOG_CUDA "cudaPointCloud::Extract() -- color width/height parameters are zero\n");
		return false;
	}

	// determine if RGB used
	if( rgba != NULL )
		mHasRGB = true;

	// upsample depth if needed
	if( mHasRGB && (depth_width != color_width || depth_height != color_height) )
	{
		if( !allocDepthResize(color_width * color_height * sizeof(float)) )
			return false;

		if( CUDA_FAILED(cudaColormap(depth, depth_width, depth_height,
							    mDepthResize, color_width, color_height,
							    make_float2(0.0f,0.0f), FORMAT_DEFAULT,
							    IMAGE_GRAY32F, COLORMAP_NONE, FILTER_LINEAR)) ) 
		{
			LogError(LOG_CUDA "cudaPointCloud::Extract() -- failed to resize depth image\n");
			return false; 
		}

		depth        = mDepthResize;
		depth_width  = color_width;
		depth_height = color_height;
	}

	// allocate point cloud memory
	const uint32_t numPoints = depth_width * depth_height;

	if( !Reserve(numPoints) )
		return false;
		
	// default calibration if needed
	if( !mHasCalibration )
	{
		const float f_w = (float)depth_width;
		const float f_h = (float)depth_height;

		mFocalLength = make_float2(f_h, f_h);
		mPrincipalPoint = make_float2(f_w * 0.5f, f_h * 0.5f);
	}

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(depth_width,blockDim.x), iDivUp(depth_height,blockDim.y));

	if( mHasRGB )
		gpuPointCloudExtract<true><<<gridDim, blockDim>>>(depth, rgba, depth_width, depth_height, mFocalLength, mPrincipalPoint, mPointsGPU);
	else
		gpuPointCloudExtract<false><<<gridDim, blockDim>>>(depth, rgba, depth_width, depth_height, mFocalLength, mPrincipalPoint, mPointsGPU);

	// check for launch errors
	if( CUDA_FAILED(cudaGetLastError()) )
	{
		LogError(LOG_CUDA "cudaPointCloud::Extract() -- failed to extra point cloud with CUDA\n");
		return false;
	}
	
	mNumPoints = numPoints;
	mHasNewPoints = true;

	return true;
}


// Extract
bool cudaPointCloud::Extract( float* depth, float4* rgba, uint32_t width, uint32_t height )
{
	return Extract(depth, width, height, rgba, width, height);
}







