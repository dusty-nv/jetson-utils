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

#include "cudaYUV.h"
#include "imageFormat.h"


//-----------------------------------------------------------------------------------
// YUV to RGB colorspace conversion
//-----------------------------------------------------------------------------------
static inline __device__ float clamp( float x )	
{ 
	return fminf(fmaxf(x, 0.0f), 255.0f); 
}

static inline __device__ float3 YUV2RGB(float Y, float U, float V)
{
	U -= 128.0f;
	V -= 128.0f;

#if 1
	return make_float3(clamp(Y + 1.4065f * V),
    				    clamp(Y - 0.3455f * U - 0.7169f * V),
				    clamp(Y + 1.7790f * U));
#else
	return make_float3(clamp(Y + 1.402f * V),
    				    clamp(Y - 0.344f * U - 0.714f * V),
				    clamp(Y + 1.772f * U));
#endif
}

//-----------------------------------------------------------------------------------
// YUYV/UYVY are macropixel formats, and two RGB pixels are output at once.
// Define vectors with 6 and 8 elements so they can be written at one time.
// These are similar to those from cudaVector.h, except for 6/8 elements.
//-----------------------------------------------------------------------------------
struct /*__align__(6)*/ uchar6
{
   uint8_t x0, y0, z0, x1, y1, z1;
};

struct __align__(8) uchar8
{
   uint8_t x0, y0, z0, w0, x1, y1, z1, w1;
};

struct /*__align__(24)*/ float6
{
   float x0, y0, z0, x1, y1, z1;
};

struct __align__(32) float8
{
   float x0, y0, z0, w0, x1, y1, z1, w1;
};

template<class T> struct vecTypeInfo;

template<> struct vecTypeInfo<uchar6> { typedef uint8_t Base; };
template<> struct vecTypeInfo<uchar8> { typedef uint8_t Base; };

template<> struct vecTypeInfo<float6> { typedef float Base; };
template<> struct vecTypeInfo<float8> { typedef float Base; };

template<typename T> struct vec_assert_false : std::false_type { };

#define BaseType typename vecTypeInfo<T>::Base

template<typename T> inline __host__ __device__ T make_vec(BaseType x0, BaseType y0, BaseType z0, BaseType w0, BaseType x1, BaseType y1, BaseType z1, BaseType w1) { static_assert(vec_assert_false<T>::value, "invalid vector type - supported types are uchar6, uchar8, float6, float8");  }

template<> inline __host__ __device__ uchar6 make_vec( uint8_t x0, uint8_t y0, uint8_t z0, uint8_t w0, uint8_t x1, uint8_t y1, uint8_t z1, uint8_t w1 )	{ return {x0, y0, z0, x1, y1, z1}; }
template<> inline __host__ __device__ uchar8 make_vec( uint8_t x0, uint8_t y0, uint8_t z0, uint8_t w0, uint8_t x1, uint8_t y1, uint8_t z1, uint8_t w1 )	{ return {x0, y0, z0, w1, x1, y1, z1, w1}; }

template<> inline __host__ __device__ float6 make_vec( float x0, float y0, float z0, float w0, float x1, float y1, float z1, float w1 )				{ return {x0, y0, z0, x1, y1, z1}; }
template<> inline __host__ __device__ float8 make_vec( float x0, float y0, float z0, float w0, float x1, float y1, float z1, float w1 )				{ return {x0, y0, z0, w1, x1, y1, z1, w1}; }


//-----------------------------------------------------------------------------------
// YUYV/UYVY to RGBA
//-----------------------------------------------------------------------------------
template <typename T, imageFormat format>
__global__ void YUYVToRGBA( uchar4* src, T* dst, int halfWidth, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= halfWidth || y >= height )
		return;

	const uchar4 macroPx = src[y * halfWidth + x];

	// Y0 is the brightness of pixel 0, Y1 the brightness of pixel 1.
	// U and V is the color of both pixels.
	float y0, y1, u, v;

	if( format == IMAGE_YUYV )
	{
		// YUYV [ Y0 | U0 | Y1 | V0 ]
		y0 = macroPx.x;
		y1 = macroPx.z;
		u  = macroPx.y;
		v  = macroPx.w;
	}
	else if( format == IMAGE_YVYU )
	{
		// YVYU [ Y0 | V0 | Y1 | U0 ]
		y0 = macroPx.x;
		y1 = macroPx.z;
		u  = macroPx.w;
		v  = macroPx.y;
	}
	else // if( format == IMAGE_UYVY )
	{
		// UYVY [ U0 | Y0 | V0 | Y1 ]
		y0 = macroPx.y;
		y1 = macroPx.w;
		u  = macroPx.x;
		v  = macroPx.z;
	}

	// this function outputs two pixels from one YUYV macropixel
	const float3 px0 = YUV2RGB(y0, u, v);
	const float3 px1 = YUV2RGB(y1, u, v);

	dst[y * halfWidth + x] = make_vec<T>(px0.x, px0.y, px0.z, 255,
								  px1.x, px1.y, px1.z, 255);
} 

template<typename T, imageFormat format>
static cudaError_t launchYUYVToRGB( void* input, T* output, size_t width, size_t height, cudaStream_t stream)
{
	if( !input || !output || !width || !height )
		return cudaErrorInvalidValue;

	const int  halfWidth = width / 2;	// two pixels are output at once
	const dim3 blockDim(8,8);
	const dim3 gridDim(iDivUp(halfWidth, blockDim.x), iDivUp(height, blockDim.y));

	YUYVToRGBA<T, format><<<gridDim, blockDim, 0, stream>>>((uchar4*)input, output, halfWidth, height);

	return CUDA(cudaGetLastError());
}


// cudaYUYVToRGB (uchar3)
cudaError_t cudaYUYVToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar6, IMAGE_YUYV>(input, (uchar6*)output, width, height, stream);
}

// cudaYUYVToRGB (float3)
cudaError_t cudaYUYVToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float6, IMAGE_YUYV>(input, (float6*)output, width, height, stream);
}

// cudaYUYVToRGBA (uchar4)
cudaError_t cudaYUYVToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar8, IMAGE_YUYV>(input, (uchar8*)output, width, height, stream);
}

// cudaYUYVToRGBA (float4)
cudaError_t cudaYUYVToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float8, IMAGE_YUYV>(input, (float8*)output, width, height, stream);
}

//-----------------------------------------------------------------------------------

// cudaUYVYToRGB (uchar3)
cudaError_t cudaUYVYToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar6, IMAGE_UYVY>(input, (uchar6*)output, width, height, stream);
}

// cudaUYVYToRGB (float3)
cudaError_t cudaUYVYToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float6, IMAGE_UYVY>(input, (float6*)output, width, height, stream);
}

// cudaUYVYToRGBA (uchar4)
cudaError_t cudaUYVYToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar8, IMAGE_UYVY>(input, (uchar8*)output, width, height, stream);
}

// cudaUYVYToRGBA (float4)
cudaError_t cudaUYVYToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float8, IMAGE_UYVY>(input, (float8*)output, width, height, stream);
}

//-----------------------------------------------------------------------------------

// cudaYVYUToRGB (uchar3)
cudaError_t cudaYVYUToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar6, IMAGE_YVYU>(input, (uchar6*)output, width, height, stream);
}

// cudaYUYVToRGB (float3)
cudaError_t cudaYVYUToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float6, IMAGE_YVYU>(input, (float6*)output, width, height, stream);
}

// cudaYUYVToRGBA (uchar4)
cudaError_t cudaYVYUToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar8, IMAGE_YVYU>(input, (uchar8*)output, width, height, stream);
}

// cudaYUYVToRGBA (float4)
cudaError_t cudaYVYUToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float8, IMAGE_YVYU>(input, (float8*)output, width, height, stream);
}

