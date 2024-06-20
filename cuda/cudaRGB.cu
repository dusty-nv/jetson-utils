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


//-----------------------------------------------------------------------------------
// RGB <-> BGR
//-----------------------------------------------------------------------------------
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
static cudaError_t launchRGBToBGR( T* srcDev, T* dstDev, size_t width, size_t height, cudaStream_t stream )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBToBGR<T><<<gridDim, blockDim, 0, stream>>>(srcDev, dstDev, width, height);
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaRGB8ToBGR8( uchar3* input, uchar3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBToBGR<uchar3>(input, output, width, height, stream);
}

cudaError_t cudaRGB32ToBGR32( float3* input, float3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBToBGR<float3>(input, output, width, height, stream);
}

cudaError_t cudaRGBA8ToBGRA8( uchar4* input, uchar4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBToBGR<uchar4>(input, output, width, height, stream);
}

cudaError_t cudaRGBA32ToBGRA32( float4* input, float4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBToBGR<float4>(input, output, width, height, stream);
}

//-----------------------------------------------------------------------------------
// uint8 to float
//-----------------------------------------------------------------------------------
template<typename T_in, typename T_out, bool isBGR>
__global__ void RGBToRGB(T_in* srcImage, T_out* dstImage, int width, int height)
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
static cudaError_t launchRGBToRGB( T_in* srcDev, T_out* dstDev, size_t width, size_t height, cudaStream_t stream )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBToRGB<T_in, T_out, isBGR><<<gridDim, blockDim, 0, stream>>>(srcDev, dstDev, width, height);
	
	return CUDA(cudaGetLastError());
}


// cudaRGB8ToRGB32 (uchar3 -> float3)
cudaError_t cudaRGB8ToRGB32( uchar3* srcDev, float3* dstDev, size_t width, size_t height, bool swapRedBlue, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB<uchar3, float3, true>(srcDev, dstDev, width, height, stream);
	else
		return launchRGBToRGB<uchar3, float3, false>(srcDev, dstDev, width, height, stream);
}

// cudaRGB8ToRGBA32 (uchar3 -> float4)
cudaError_t cudaRGB8ToRGBA32( uchar3* srcDev, float4* dstDev, size_t width, size_t height, bool swapRedBlue, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB<uchar3, float4, true>(srcDev, dstDev, width, height, stream);
	else
		return launchRGBToRGB<uchar3, float4, false>(srcDev, dstDev, width, height, stream);
}

// cudaRGBA8ToRGB32 (uchar4 -> float3)
cudaError_t cudaRGBA8ToRGB32( uchar4* srcDev, float3* dstDev, size_t width, size_t height, bool swapRedBlue, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB<uchar4, float3, true>(srcDev, dstDev, width, height, stream);
	else
		return launchRGBToRGB<uchar4, float3, false>(srcDev, dstDev, width, height, stream);
}

// cudaRGBA8ToRGBA32 (uchar4 -> float4)
cudaError_t cudaRGBA8ToRGBA32( uchar4* srcDev, float4* dstDev, size_t width, size_t height, bool swapRedBlue, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB<uchar4, float4, true>(srcDev, dstDev, width, height, stream);
	else
		return launchRGBToRGB<uchar4, float4, false>(srcDev, dstDev, width, height, stream);
}

// cudaRGB8ToRGBA8 (uchar3 -> uchar4)
cudaError_t cudaRGB8ToRGBA8( uchar3* srcDev, uchar4* dstDev, size_t width, size_t height, bool swapRedBlue, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB<uchar3, uchar4, true>(srcDev, dstDev, width, height, stream);
	else
		return launchRGBToRGB<uchar3, uchar4, false>(srcDev, dstDev, width, height, stream);
}

// cudaRGBA8ToRGB8 (uchar4 -> uchar3)
cudaError_t cudaRGBA8ToRGB8( uchar4* srcDev, uchar3* dstDev, size_t width, size_t height, bool swapRedBlue, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB<uchar4, uchar3, true>(srcDev, dstDev, width, height, stream);
	else
		return launchRGBToRGB<uchar4, uchar3, false>(srcDev, dstDev, width, height, stream);
}

// cudaRGB32ToRGBA32 (float3 -> float4)
cudaError_t cudaRGB32ToRGBA32( float3* srcDev, float4* dstDev, size_t width, size_t height, bool swapRedBlue, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB<float3, float4, true>(srcDev, dstDev, width, height, stream);
	else
		return launchRGBToRGB<float3, float4, false>(srcDev, dstDev, width, height, stream);
}

// cudaRGBA32ToRGB32 (float4 -> float3)
cudaError_t cudaRGBA32ToRGB32( float4* srcDev, float3* dstDev, size_t width, size_t height, bool swapRedBlue, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB<float4, float3, true>(srcDev, dstDev, width, height, stream);
	else
		return launchRGBToRGB<float4, float3, false>(srcDev, dstDev, width, height, stream);
}

//-----------------------------------------------------------------------------------
// float to uint8
//-----------------------------------------------------------------------------------
template<typename T_in, typename T_out, bool isBGRA>
__global__ void RGBToRGB_Norm(T_in* srcImage, T_out* dstImage, int width, int height,
						float2 input_range, float scaling_factor)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	const T_in px = srcImage[pixel];

	#define rescale(v) ((v - input_range.x) * scaling_factor)

	if( isBGRA )
		dstImage[pixel] = make_vec<T_out>(rescale(px.z), rescale(px.y), rescale(px.x), rescale(alpha(px,input_range.y)));
	else
		dstImage[pixel] = make_vec<T_out>(rescale(px.x), rescale(px.y), rescale(px.z), rescale(alpha(px,input_range.y)));
}

template<typename T_in, typename T_out, bool isBGR> 
static cudaError_t launchRGBToRGB_Norm( T_in* srcDev, T_out* dstDev, size_t width, size_t height, const float2& inputRange, cudaStream_t stream )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const float multiplier = 255.0f / (inputRange.y - inputRange.x);

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBToRGB_Norm<T_in, T_out, isBGR><<<gridDim, blockDim, 0, stream>>>( srcDev, dstDev, width, height, inputRange, multiplier);
	
	return CUDA(cudaGetLastError());
}


// cudaRGB32ToRGB8 (float3 -> uchar3)
cudaError_t cudaRGB32ToRGB8( float3* srcDev, uchar3* dstDev, size_t width, size_t height, bool swapRedBlue, const float2& inputRange, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB_Norm<float3, uchar3, true>(srcDev, dstDev, width, height, inputRange, stream);
	else
		return launchRGBToRGB_Norm<float3, uchar3, false>(srcDev, dstDev, width, height, inputRange, stream);
}

// cudaRGB32ToRGBA8 (float3 -> uchar4)
cudaError_t cudaRGB32ToRGBA8( float3* srcDev, uchar4* dstDev, size_t width, size_t height, bool swapRedBlue, const float2& inputRange, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB_Norm<float3, uchar4, true>(srcDev, dstDev, width, height, inputRange, stream);
	else
		return launchRGBToRGB_Norm<float3, uchar4, false>(srcDev, dstDev, width, height, inputRange, stream);
}

// cudaRGBA32ToRGB8 (float4 -> uchar3)
cudaError_t cudaRGBA32ToRGB8( float4* srcDev, uchar3* dstDev, size_t width, size_t height, bool swapRedBlue, const float2& inputRange, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB_Norm<float4, uchar3, true>(srcDev, dstDev, width, height, inputRange, stream);
	else
		return launchRGBToRGB_Norm<float4, uchar3, false>(srcDev, dstDev, width, height, inputRange, stream);
}

// cudaRGBA32ToRGBA8 (float4 -> uchar4)
cudaError_t cudaRGBA32ToRGBA8( float4* srcDev, uchar4* dstDev, size_t width, size_t height, bool swapRedBlue, const float2& inputRange, cudaStream_t stream )
{
	if( swapRedBlue )
		return launchRGBToRGB_Norm<float4, uchar4, true>(srcDev, dstDev, width, height, inputRange, stream);
	else
		return launchRGBToRGB_Norm<float4, uchar4, false>(srcDev, dstDev, width, height, inputRange, stream);
}
