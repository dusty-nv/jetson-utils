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


inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}


/* From RGB to YUV

   Y = 0.299R + 0.587G + 0.114B
   U = 0.492 (B-Y)
   V = 0.877 (R-Y)

   It can also be represented as:

   Y =  0.299R + 0.587G + 0.114B
   U = -0.147R - 0.289G + 0.436B
   V =  0.615R - 0.515G - 0.100B

   From YUV to RGB

   R = Y + 1.140V
   G = Y - 0.395U - 0.581V
   B = Y + 2.032U
 */

struct __align__(8) uchar8
{
   uint8_t a0, a1, a2, a3, a4, a5, a6, a7;
};

struct __align__(32) float8
{
   float a0, a1, a2, a3, a4, a5, a6, a7;
};

template<typename T, typename BaseT> __device__ __forceinline__ T make_vec8(BaseT a0, BaseT a1, BaseT a2, BaseT a3, BaseT a4, BaseT a5, BaseT a6, BaseT a7)
{
   T val = {a0, a1, a2, a3, a4, a5, a6, a7};
   return val;
}

/*static __host__ __device__ __forceinline__ uchar8 make_uchar8(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4, uint8_t a5, uint8_t a6, uint8_t a7)
{
   uchar8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
   return val;
}


static __host__ __device__ __forceinline__ float8 make_float8(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7)
{
   float8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
   return val;
}*/


//-----------------------------------------------------------------------------------
// YUYV/UYVY to RGBA
//-----------------------------------------------------------------------------------
template <typename T, typename BaseT, bool formatUYVY>
__global__ void yuyvToRgba( uchar4* src, int srcAlignedWidth, T* dst, int dstAlignedWidth, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= srcAlignedWidth || y >= height )
		return;

	const uchar4 macroPx = src[y * srcAlignedWidth + x];

	// Y0 is the brightness of pixel 0, Y1 the brightness of pixel 1.
	// U0 and V0 is the color of both pixels.
	// UYVY [ U0 | Y0 | V0 | Y1 ] 
	// YUYV [ Y0 | U0 | Y1 | V0 ]
	const float y0 = formatUYVY ? macroPx.y : macroPx.x;
	const float y1 = formatUYVY ? macroPx.w : macroPx.z; 
	const float u = (formatUYVY ? macroPx.x : macroPx.y) - 128.0f;
	const float v = (formatUYVY ? macroPx.z : macroPx.w) - 128.0f;

	const float4 px0 = make_float4( y0 + 1.4065f * v,
							  y0 - 0.3455f * u - 0.7169f * v,
							  y0 + 1.7790f * u, 255.0f );

	const float4 px1 = make_float4( y1 + 1.4065f * v,
							  y1 - 0.3455f * u - 0.7169f * v,
							  y1 + 1.7790f * u, 255.0f );

	dst[y * dstAlignedWidth + x] = make_vec8<T, BaseT>(clamp(px0.x, 0.0f, 255.0f), 
									    clamp(px0.y, 0.0f, 255.0f),
									    clamp(px0.z, 0.0f, 255.0f),
									    clamp(px0.w, 0.0f, 255.0f),
									    clamp(px1.x, 0.0f, 255.0f),
									    clamp(px1.y, 0.0f, 255.0f),
									    clamp(px1.z, 0.0f, 255.0f),
									    clamp(px1.w, 0.0f, 255.0f));
} 

template<typename T, typename BaseT, bool formatUYVY>
cudaError_t launchYUYV( void* input, size_t inputPitch, T* output, size_t outputPitch, size_t width, size_t height)
{
	if( !input || !inputPitch || !output || !outputPitch || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(8,8);
	const dim3 grid(iDivUp(width/2, block.x), iDivUp(height, block.y));

	const int srcAlignedWidth = inputPitch / sizeof(uchar4);	// normally would be uchar2, but we're doubling up pixels
	const int dstAlignedWidth = outputPitch / sizeof(T);	// normally would be uchar4 ^^^

	//printf("yuyvToRgba %zu %zu %i %i %i %i %i\n", width, height, (int)formatUYVY, srcAlignedWidth, dstAlignedWidth, grid.x, grid.y);

	yuyvToRgba<T, BaseT, formatUYVY><<<grid, block>>>((uchar4*)input, srcAlignedWidth, output, dstAlignedWidth, width, height);

	return CUDA(cudaGetLastError());
}


// cudaUYVYToRGBA (uchar4)
cudaError_t cudaUYVYToRGBA( void* input, uchar4* output, size_t width, size_t height )
{
	return cudaUYVYToRGBA(input, width * sizeof(uchar2), output, width * sizeof(uchar4), width, height);
}

// cudaUYVYToRGBA (uchar4)
cudaError_t cudaUYVYToRGBA( void* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height )
{
	return launchYUYV<uchar8, uint8_t, true>(input, inputPitch, (uchar8*)output, outputPitch, width, height);
}

// cudaYUYVToRGBA (uchar4)
cudaError_t cudaYUYVToRGBA( void* input, uchar4* output, size_t width, size_t height )
{
	return cudaYUYVToRGBA(input, width * sizeof(uchar2), output, width * sizeof(uchar4), width, height);
}

// cudaYUYVToRGBA (uchar4)
cudaError_t cudaYUYVToRGBA( void* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height )
{
	return launchYUYV<uchar8, uint8_t, false>(input, inputPitch, (uchar8*)output, outputPitch, width, height);
}

// cudaUYVYToRGBA (float4)
cudaError_t cudaUYVYToRGBA( void* input, float4* output, size_t width, size_t height )
{
	return cudaUYVYToRGBA(input, width * sizeof(uchar2), output, width * sizeof(float4), width, height);
}

// cudaUYVYToRGBA (float4)
cudaError_t cudaUYVYToRGBA( void* input, size_t inputPitch, float4* output, size_t outputPitch, size_t width, size_t height )
{
	return launchYUYV<float8, float, true>(input, inputPitch, (float8*)output, outputPitch, width, height);
}

// cudaYUYVToRGBA (float4)
cudaError_t cudaYUYVToRGBA( void* input, float4* output, size_t width, size_t height )
{
	return cudaYUYVToRGBA(input, width * sizeof(uchar2), output, width * sizeof(float4), width, height);
}

// cudaYUYVToRGBA (float4)
cudaError_t cudaYUYVToRGBA( void* input, size_t inputPitch, float4* output, size_t outputPitch, size_t width, size_t height )
{
	return launchYUYV<float8, float, false>(input, inputPitch, (float8*)output, outputPitch, width, height);
}


//-----------------------------------------------------------------------------------
// YUYV/UYVY to grayscale
//-----------------------------------------------------------------------------------

template <bool formatUYVY>
__global__ void yuyvToGray( uchar4* src, int srcAlignedWidth, float2* dst, int dstAlignedWidth, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= srcAlignedWidth || y >= height )
		return;

	const uchar4 macroPx = src[y * srcAlignedWidth + x];

	const float y0 = formatUYVY ? macroPx.y : macroPx.x;
	const float y1 = formatUYVY ? macroPx.w : macroPx.z; 

	dst[y * dstAlignedWidth + x] = make_float2(y0/255.0f, y1/255.0f);
} 

template<bool formatUYVY>
cudaError_t launchGrayYUYV( void* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height)
{
	if( !input || !inputPitch || !output || !outputPitch || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(8,8);
	const dim3 grid(iDivUp(width/2, block.x), iDivUp(height, block.y));

	const int srcAlignedWidth = inputPitch / sizeof(uchar4);	// normally would be uchar2, but we're doubling up pixels
	const int dstAlignedWidth = outputPitch / sizeof(float2);	// normally would be float ^^^

	yuyvToGray<formatUYVY><<<grid, block>>>((uchar4*)input, srcAlignedWidth, (float2*)output, dstAlignedWidth, width, height);

	return CUDA(cudaGetLastError());
}

cudaError_t cudaUYVYToGray( void* input, float* output, size_t width, size_t height )
{
	return cudaUYVYToGray(input, width * sizeof(uchar2), output, width * sizeof(uint8_t), width, height);
}

cudaError_t cudaUYVYToGray( void* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height )
{
	return launchGrayYUYV<true>(input, inputPitch, output, outputPitch, width, height);
}

cudaError_t cudaYUYVToGray( void* input, float* output, size_t width, size_t height )
{
	return cudaYUYVToGray(input, width * sizeof(uchar2), output, width * sizeof(float), width, height);
}

cudaError_t cudaYUYVToGray( void* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height )
{
	return launchGrayYUYV<false>(input, inputPitch, output, outputPitch, width, height);
}

