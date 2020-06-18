/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "cudaGrayscale.h"
#include "cudaVector.h"


//-----------------------------------------------------------------------------------
// RGB to Grayscale
//-----------------------------------------------------------------------------------
inline __device__ float RGB2Gray( float3 rgb )
{
	return float(rgb.x) * 0.2989f + float(rgb.y) * 0.5870f + float(rgb.z) * 0.1140f;
}

template<typename T_in, typename T_out, bool isBGR>
__global__ void RGBToGray(T_in* srcImage, T_out* dstImage, int width, int height)
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
		dstImage[pixel] = RGB2Gray(make_float3(px.z, px.y, px.x));
	else
		dstImage[pixel] = RGB2Gray(make_float3(px.x, px.y, px.z));
}

template<typename T_in, typename T_out, bool isBGR> 
static cudaError_t launchRGBToGray( T_in* srcDev, T_out* dstDev, size_t width, size_t height )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBToGray<T_in, T_out, isBGR><<<gridDim, blockDim>>>( srcDev, dstDev, width, height );
	
	return CUDA(cudaGetLastError());
}

// cudaRGB8ToGray8 (uchar3 -> uint8)
cudaError_t cudaRGB8ToGray8( uchar3* srcDev, uint8_t* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGBToGray<uchar3, uint8_t, true>(srcDev, dstDev, width, height);
	else
		return launchRGBToGray<uchar3, uint8_t, false>(srcDev, dstDev, width, height);
}

// cudaRGBA8ToGray8 (uchar4 -> uint8)
cudaError_t cudaRGBA8ToGray8( uchar4* srcDev, uint8_t* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGBToGray<uchar4, uint8_t, true>(srcDev, dstDev, width, height);
	else
		return launchRGBToGray<uchar4, uint8_t, false>(srcDev, dstDev, width, height);
}

// cudaRGB8ToGray32 (uchar3 -> float)
cudaError_t cudaRGB8ToGray32( uchar3* srcDev, float* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGBToGray<uchar3, float, true>(srcDev, dstDev, width, height);
	else
		return launchRGBToGray<uchar3, float, false>(srcDev, dstDev, width, height);
}

// cudaRGBA8ToGray32 (uchar4 -> float)
cudaError_t cudaRGBA8ToGray32( uchar4* srcDev, float* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGBToGray<uchar4, float, true>(srcDev, dstDev, width, height);
	else
		return launchRGBToGray<uchar4, float, false>(srcDev, dstDev, width, height);
}

// cudaRGB32ToGray32 (float3 -> float)
cudaError_t cudaRGB32ToGray32( float3* srcDev, float* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGBToGray<float3, float, true>(srcDev, dstDev, width, height);
	else
		return launchRGBToGray<float3, float, false>(srcDev, dstDev, width, height);
}

// cudaRGBA32ToGray32 (float4 -> float)
cudaError_t cudaRGBA32ToGray32( float4* srcDev, float* dstDev, size_t width, size_t height, bool swapRedBlue )
{
	if( swapRedBlue )
		return launchRGBToGray<float4, float, true>(srcDev, dstDev, width, height);
	else
		return launchRGBToGray<float4, float, false>(srcDev, dstDev, width, height);
}


//-----------------------------------------------------------------------------------
// RGB to Grayscale (normalized)
//-----------------------------------------------------------------------------------
template<typename T_in, typename T_out, bool isBGR>
__global__ void RGBToGray_Norm(T_in* srcImage, T_out* dstImage, int width, int height,
							   float min_pixel_value, float scaling_factor)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	#define rescale(v) ((v - min_pixel_value) * scaling_factor)

	const T_in px = srcImage[pixel];

	if( isBGR )
		dstImage[pixel] = RGB2Gray(make_float3(rescale(px.z), rescale(px.y), rescale(px.x)));
	else
		dstImage[pixel] = RGB2Gray(make_float3(rescale(px.x), rescale(px.y), rescale(px.z)));
}

template<typename T_in, typename T_out, bool isBGR> 
static cudaError_t launchRGBToGray_Norm( T_in* srcDev, T_out* dstDev, size_t width, size_t height, const float2& inputRange )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const float multiplier = 255.0f / (inputRange.y - inputRange.x);

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBToGray_Norm<T_in, T_out, isBGR><<<gridDim, blockDim>>>( srcDev, dstDev, width, height, inputRange.x, multiplier );
	
	return CUDA(cudaGetLastError());
}

// cudaRGB32ToGray8 (float3 -> uint8)
cudaError_t cudaRGB32ToGray8( float3* srcDev, uint8_t* dstDev, size_t width, size_t height, bool swapRedBlue, const float2& inputRange )
{
	if( swapRedBlue )
		return launchRGBToGray_Norm<float3, uint8_t, true>(srcDev, dstDev, width, height, inputRange);
	else
		return launchRGBToGray_Norm<float3, uint8_t, false>(srcDev, dstDev, width, height, inputRange);
}

// cudaRGBA32ToGray8 (float4 -> uint8)
cudaError_t cudaRGBA32ToGray8( float4* srcDev, uint8_t* dstDev, size_t width, size_t height, bool swapRedBlue, const float2& inputRange )
{
	if( swapRedBlue )
		return launchRGBToGray_Norm<float4, uint8_t, true>(srcDev, dstDev, width, height, inputRange);
	else
		return launchRGBToGray_Norm<float4, uint8_t, false>(srcDev, dstDev, width, height, inputRange);
}


//-----------------------------------------------------------------------------------
// Grayscale to RGB
//-----------------------------------------------------------------------------------
template<typename T_in, typename T_out>
__global__ void GrayToRGB(T_in* srcImage, T_out* dstImage, int width, int height)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	const T_in px = srcImage[pixel];
	dstImage[pixel] = make_vec<T_out>(px, px, px, 255);
}

template<typename T_in, typename T_out> 
cudaError_t launchGrayToRGB( T_in* srcDev, T_out* dstDev, size_t width, size_t height )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	GrayToRGB<T_in, T_out><<<gridDim, blockDim>>>( srcDev, dstDev, width, height );
	
	return CUDA(cudaGetLastError());
}

// cudaGray8ToRGB8 (uint8 -> uchar3)
cudaError_t cudaGray8ToRGB8( uint8_t* srcDev, uchar3* dstDev, size_t width, size_t height )
{
	return launchGrayToRGB<uint8_t, uchar3>(srcDev, dstDev, width, height);
}

// cudaGray8ToRGBA8 (uint8 -> uchar4)
cudaError_t cudaGray8ToRGBA8( uint8_t* srcDev, uchar4* dstDev, size_t width, size_t height )
{
	return launchGrayToRGB<uint8_t, uchar4>(srcDev, dstDev, width, height);
}

// cudaGray8ToRGB32 (uint8 -> float3)
cudaError_t cudaGray8ToRGB32( uint8_t* srcDev, float3* dstDev, size_t width, size_t height )
{
	return launchGrayToRGB<uint8_t, float3>(srcDev, dstDev, width, height);
}

// cudaGray8ToRGBA32 (uint8 -> float4)
cudaError_t cudaGray8ToRGBA32( uint8_t* srcDev, float4* dstDev, size_t width, size_t height )
{
	return launchGrayToRGB<uint8_t, float4>(srcDev, dstDev, width, height);
}

// cudaGray32ToRGB32 (float -> float3)
cudaError_t cudaGray32ToRGB32( float* srcDev, float3* dstDev, size_t width, size_t height )
{
	return launchGrayToRGB<float, float3>(srcDev, dstDev, width, height);
}

// cudaGray32ToRGBA32 (float -> float4)
cudaError_t cudaGray32ToRGBA32( float* srcDev, float4* dstDev, size_t width, size_t height )
{
	return launchGrayToRGB<float, float4>(srcDev, dstDev, width, height);
}

// cudaGray8ToGray32 (uint8 -> float)
cudaError_t cudaGray8ToGray32( uint8_t* srcDev, float* dstDev, size_t width, size_t height )
{
	return launchGrayToRGB<uint8_t, float>(srcDev, dstDev, width, height);
}


//-----------------------------------------------------------------------------------
// Grayscale to RGB (normalized)
//-----------------------------------------------------------------------------------
template<typename T_in, typename T_out>
__global__ void GrayToRGB_Norm(T_in* srcImage, T_out* dstImage, int width, int height,
							   float min_pixel_value, float scaling_factor)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	const T_in px   = rescale(srcImage[pixel]);
	dstImage[pixel] = make_vec<T_out>(px, px, px, 255);
}

template<typename T_in, typename T_out> 
static cudaError_t launchGrayToRGB_Norm( T_in* srcDev, T_out* dstDev, size_t width, size_t height, const float2& inputRange )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const float multiplier = 255.0f / (inputRange.y - inputRange.x);

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	GrayToRGB_Norm<T_in, T_out><<<gridDim, blockDim>>>( srcDev, dstDev, width, height, inputRange.x, multiplier );
	
	return CUDA(cudaGetLastError());
}

// cudaGray32ToRGB8 (float-> uchar3)
cudaError_t cudaGray32ToRGB8( float* srcDev, uchar3* dstDev, size_t width, size_t height, const float2& inputRange )
{
	return launchGrayToRGB_Norm<float, uchar3>(srcDev, dstDev, width, height, inputRange);
}

// cudaGray32ToRGBA8 (float-> uchar4)
cudaError_t cudaGray32ToRGBA8( float* srcDev, uchar4* dstDev, size_t width, size_t height, const float2& inputRange )
{
	return launchGrayToRGB_Norm<float, uchar4>(srcDev, dstDev, width, height, inputRange);
}

// cudaGray32ToGray8 (float -> uint8)
cudaError_t cudaGray32ToGray8( float* srcDev, uint8_t* dstDev, size_t width, size_t height, const float2& inputRange )
{
	return launchGrayToRGB_Norm<float, uint8_t>(srcDev, dstDev, width, height, inputRange);
}



