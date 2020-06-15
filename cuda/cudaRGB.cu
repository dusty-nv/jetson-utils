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
#include "cudaVector.h"


//-------------------------------------------------------------------------------------------------------------------------
template<typename T>
__global__ void RGBToBGR(T* srcImage, T* dstImage, int width, int height)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	const T px = srcImage[pixel];
	
	dstImage[pixel] = make_vec<T>(px.z, px.y, px.x, alpha(px));
}

template<typename T> 
cudaError_t launchRGBToBGR( T* srcDev, T* dstDev, size_t width, size_t height )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBToBGR<T><<<gridDim, blockDim>>>(srcDev, dstDev, width, height);
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaRGB8ToBGR8( uchar3* input, uchar3* output, size_t width, size_t height )
{
	return launchRGBToBGR<uchar3>(input, output, width, height);
}

cudaError_t cudaRGB32ToBGR32( float3* input, float3* output, size_t width, size_t height )
{
	return launchRGBToBGR<float3>(input, output, width, height);
}

cudaError_t cudaRGBA8ToBGRA8( uchar4* input, uchar4* output, size_t width, size_t height )
{
	return launchRGBToBGR<uchar4>(input, output, width, height);
}

cudaError_t cudaRGBA32ToBGRA32( float4* input, float4* output, size_t width, size_t height )
{
	return launchRGBToBGR<float4>(input, output, width, height);
}

//-------------------------------------------------------------------------------------------------------------------------
template<typename T_in, typename T_out, bool isBGR>
__global__ void RGB8ToRGB32(T_in* srcImage, T_out* dstImage, int width, int height)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	const T_in px = srcImage[pixel];
	
	if( isBGR )
		dstImage[pixel] = make_vec<T_out>(px.z, px.y, px.x, alpha(px));
	else
		dstImage[pixel] = make_vec<T_out>(px.x, px.y, px.z, alpha(px));
}

template<typename T_in, typename T_out, bool isBGR> 
cudaError_t launchRGB8ToRGB32( T_in* srcDev, T_out* dstDev, size_t width, size_t height )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGB8ToRGB32<T_in, T_out, isBGR><<<gridDim, blockDim>>>(srcDev, dstDev, width, height);
	
	return CUDA(cudaGetLastError());
}


// cudaRGB8ToRGB32 (uchar3 -> float3)
cudaError_t cudaRGB8ToRGB32( uchar3* srcDev, float3* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGB8ToRGB32<uchar3, float3, true>(srcDev, dstDev, width, height);
	else
		return launchRGB8ToRGB32<uchar3, float3, false>(srcDev, dstDev, width, height);
}

// cudaRGB8ToRGBA32 (uchar3 -> float4)
cudaError_t cudaRGB8ToRGBA32( uchar3* srcDev, float4* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGB8ToRGB32<uchar3, float4, true>(srcDev, dstDev, width, height);
	else
		return launchRGB8ToRGB32<uchar3, float4, false>(srcDev, dstDev, width, height);
}

// cudaRGBA8ToRGB32 (uchar4 -> float3)
cudaError_t cudaRGBA8ToRGB32( uchar4* srcDev, float3* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGB8ToRGB32<uchar4, float3, true>(srcDev, dstDev, width, height);
	else
		return launchRGB8ToRGB32<uchar4, float3, false>(srcDev, dstDev, width, height);
}

// cudaRGBA8ToRGBA32 (uchar4 -> float4)
cudaError_t cudaRGBA8ToRGBA32( uchar4* srcDev, float4* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGB8ToRGB32<uchar4, float4, true>(srcDev, dstDev, width, height);
	else
		return launchRGB8ToRGB32<uchar4, float4, false>(srcDev, dstDev, width, height);
}

// cudaRGB8ToRGBA8 (uchar3 -> uchar4)
cudaError_t cudaRGB8ToRGBA8( uchar3* srcDev, uchar4* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGB8ToRGB32<uchar3, uchar4, true>(srcDev, dstDev, width, height);
	else
		return launchRGB8ToRGB32<uchar3, uchar4, false>(srcDev, dstDev, width, height);
}

// cudaRGBA8ToRGB8 (uchar4 -> uchar3)
cudaError_t cudaRGBA8ToRGB8( uchar4* srcDev, uchar3* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGB8ToRGB32<uchar4, uchar3, true>(srcDev, dstDev, width, height);
	else
		return launchRGB8ToRGB32<uchar4, uchar3, false>(srcDev, dstDev, width, height);
}

// cudaRGB32ToRGBA32 (float3 -> float4)
cudaError_t cudaRGB32ToRGBA32( float3* srcDev, float4* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGB8ToRGB32<float3, float4, true>(srcDev, dstDev, width, height);
	else
		return launchRGB8ToRGB32<float3, float4, false>(srcDev, dstDev, width, height);
}

// cudaRGBA32ToRGB32 (float4 -> float3)
cudaError_t cudaRGBA32ToRGB32( float4* srcDev, float3* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGB8ToRGB32<float4, float3, true>(srcDev, dstDev, width, height);
	else
		return launchRGB8ToRGB32<float4, float3, false>(srcDev, dstDev, width, height);
}

//-------------------------------------------------------------------------------------------------------------------------
template<typename T_in, typename T_out, bool isBGRA>
__global__ void RGB32ToRGB8(T_in* srcImage, T_out* dstImage, int width, int height,
					   float min_pixel_value, float scaling_factor)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	const T_in px = srcImage[pixel];

	#define rescale(x) ((x - min_pixel_value) * scaling_factor)

	if( isBGRA )
		dstImage[pixel] = make_vec<T_out>(rescale(px.z), rescale(px.y), rescale(px.x), alpha(px));
	else
		dstImage[pixel] = make_vec<T_out>(rescale(px.x), rescale(px.y), rescale(px.z), alpha(px));
}

template<typename T_in, typename T_out, bool isBGR> 
cudaError_t launchRGB32ToRGB8( T_in* srcDev, T_out* dstDev, size_t width, size_t height, const float2& inputRange )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const float multiplier = 255.0f / (inputRange.y - inputRange.x);

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGB32ToRGB8<T_in, T_out, isBGR><<<gridDim, blockDim>>>( srcDev, dstDev, width, height, inputRange.x, multiplier);
	
	return CUDA(cudaGetLastError());
}


// cudaRGB32ToRGB8 (float3 -> uchar3)
cudaError_t cudaRGB32ToRGB8( float3* srcDev, uchar3* dstDev, size_t width, size_t height, bool swapRedBlue, const float2& inputRange )
{
	if( swapRedBlue )
		return launchRGB32ToRGB8<float3, uchar3, true>(srcDev, dstDev, width, height, inputRange);
	else
		return launchRGB32ToRGB8<float3, uchar3, false>(srcDev, dstDev, width, height, inputRange);
}

// cudaRGB32ToRGBA8 (float3 -> uchar4)
cudaError_t cudaRGB32ToRGBA8( float3* srcDev, uchar4* dstDev, size_t width, size_t height, bool swapRedBlue, const float2& inputRange )
{
	if( swapRedBlue )
		return launchRGB32ToRGB8<float3, uchar4, true>(srcDev, dstDev, width, height, inputRange);
	else
		return launchRGB32ToRGB8<float3, uchar4, false>(srcDev, dstDev, width, height, inputRange);
}

// cudaRGBA32ToRGB8 (float4 -> uchar3)
cudaError_t cudaRGBA32ToRGB8( float4* srcDev, uchar3* dstDev, size_t width, size_t height, bool swapRedBlue, const float2& inputRange )
{
	if( swapRedBlue )
		return launchRGB32ToRGB8<float4, uchar3, true>(srcDev, dstDev, width, height, inputRange);
	else
		return launchRGB32ToRGB8<float4, uchar3, false>(srcDev, dstDev, width, height, inputRange);
}

// cudaRGBA32ToRGBA8 (float4 -> uchar4)
cudaError_t cudaRGBA32ToRGBA8( float4* srcDev, uchar4* dstDev, size_t width, size_t height, bool swapRedBlue, const float2& inputRange )
{
	if( swapRedBlue )
		return launchRGB32ToRGB8<float4, uchar4, true>(srcDev, dstDev, width, height, inputRange);
	else
		return launchRGB32ToRGB8<float4, uchar4, false>(srcDev, dstDev, width, height, inputRange);
}


