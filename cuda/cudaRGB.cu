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

#include "cudaRGB.h"

//-------------------------------------------------------------------------------------------------------------------------
template<bool isBGR>
__global__ void RGBToRGBAf(uchar3* srcImage,
                           float4* dstImage,
                           int width, int height)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

//	printf("cuda thread %i %i  %i %i pixel %i \n", x, y, width, height, pixel);
		
	const float  s  = 1.0f;
	const uchar3 px = srcImage[pixel];
	
	if( isBGR )
		dstImage[pixel] = make_float4(px.z * s, px.y * s, px.x * s, 255.0f * s);
	else
		dstImage[pixel] = make_float4(px.x * s, px.y * s, px.z * s, 255.0f * s);
}

cudaError_t cudaBGR8ToRGBA32( uchar3* srcDev, float4* destDev, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBToRGBAf<true><<<gridDim, blockDim>>>( srcDev, destDev, width, height );
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaRGB8ToRGBA32( uchar3* srcDev, float4* destDev, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBToRGBAf<false><<<gridDim, blockDim>>>( srcDev, destDev, width, height );
	
	return CUDA(cudaGetLastError());
}

//-------------------------------------------------------------------------------------------------------------------------
template<bool isBGRA>
__global__ void RGBAToRGBA8(float4* srcImage,
                            uchar4* dstImage,
                            int width, int height,
					   float scaling_factor)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	const float4 px = srcImage[pixel];

	if( isBGRA )
	{
		dstImage[pixel] = make_uchar4(px.z * scaling_factor, 
								px.y * scaling_factor, 
								px.x * scaling_factor,
								px.w * scaling_factor);
	}
	else
	{
		dstImage[pixel] = make_uchar4(px.x * scaling_factor, 
								px.y * scaling_factor, 
								px.z * scaling_factor,
								px.w * scaling_factor);
	}
}

cudaError_t cudaRGBA32ToRGBA8( float4* srcDev, uchar4* destDev, size_t width, size_t height, const float2& inputRange )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const float multiplier = 255.0f / inputRange.y;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBAToRGBA8<false><<<gridDim, blockDim>>>( srcDev, destDev, width, height, multiplier );
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaRGBA32ToRGBA8( float4* srcDev, uchar4* destDev, size_t width, size_t height )
{
	return cudaRGBA32ToRGBA8(srcDev, destDev, width, height, make_float2(0.0f, 255.0f));
}

cudaError_t cudaRGBA32ToBGRA8( float4* srcDev, uchar4* destDev, size_t width, size_t height, const float2& inputRange )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const float multiplier = 255.0f / inputRange.y;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBAToRGBA8<true><<<gridDim, blockDim>>>( srcDev, destDev, width, height, multiplier );
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaRGBA32ToBGRA8( float4* srcDev, uchar4* destDev, size_t width, size_t height )
{
	return cudaRGBA32ToBGRA8(srcDev, destDev, width, height, make_float2(0.0f, 255.0f));
}


//-------------------------------------------------------------------------------------------------------------------------
template<bool isBGR>
__global__ void RGBAToRGB8(float4* srcImage,
                           uchar3* dstImage,
                           int width, int height,
					  float scaling_factor)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	const float4 px = srcImage[pixel];

	if( isBGR )
	{
		dstImage[pixel] = make_uchar3(px.z * scaling_factor, 
								px.y * scaling_factor, 
								px.x * scaling_factor);
	}
	else
	{
		dstImage[pixel] = make_uchar3(px.x * scaling_factor, 
								px.y * scaling_factor, 
								px.z * scaling_factor);
	}
}

cudaError_t cudaRGBA32ToRGB8( float4* srcDev, uchar3* destDev, size_t width, size_t height, const float2& inputRange )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const float multiplier = 255.0f / inputRange.y;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBAToRGB8<false><<<gridDim, blockDim>>>( srcDev, destDev, width, height, multiplier );
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaRGBA32ToRGB8( float4* srcDev, uchar3* destDev, size_t width, size_t height )
{
	return cudaRGBA32ToRGB8(srcDev, destDev, width, height, make_float2(0.0f, 255.0f));
}

cudaError_t cudaRGBA32ToBGR8( float4* srcDev, uchar3* destDev, size_t width, size_t height, const float2& inputRange )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const float multiplier = 255.0f / inputRange.y;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBAToRGB8<true><<<gridDim, blockDim>>>( srcDev, destDev, width, height, multiplier );
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaRGBA32ToBGR8( float4* srcDev, uchar3* destDev, size_t width, size_t height )
{
	return cudaRGBA32ToBGR8(srcDev, destDev, width, height, make_float2(0.0f, 255.0f));
}



