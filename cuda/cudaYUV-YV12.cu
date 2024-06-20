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
#include "cudaVector.h"



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
	return make_float3(clamp(Y + 1.402f * V),
    				    clamp(Y - 0.344f * U - 0.714f * V),
				    clamp(Y + 1.772f * U));
#else
	return make_float3(clamp(Y + 1.140f * V),
			    	    clamp(Y - 0.395f * U - 0.581f * V),
			         clamp(Y + 2.3032f * U));
#endif
}

//-------------------------------------------------------------------------------------
// I420/YV12 to RGB
//-------------------------------------------------------------------------------------
template <typename T, bool formatYV12>
__global__ void I420ToRGB(uint8_t* srcImage, int srcPitch,
                          T* dstImage,     	int dstPitch,
                          int width,         int height) 
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width )
		return;

	if( y >= height )
		return;

	const int x2 = x/2;
	const int y2 = y/2;

	const int srcPitch2 = srcPitch/2;
	const int planeSize = srcPitch * height;

	// get the YUV plane offsets
	uint8_t* y_plane = srcImage;
	uint8_t* u_plane;
	uint8_t* v_plane;

	if( formatYV12 )
	{
		v_plane = y_plane + planeSize;		
		u_plane = v_plane + (planeSize / 4);	// size of U & V planes is 25% of Y plane
	}
	else
	{
		u_plane = y_plane + planeSize;		// in I420, order of U & V planes is reversed
		v_plane = u_plane + (planeSize / 4);
	}

	// read YUV pixel
	const float Y = y_plane[y * srcPitch + x];
	const float U = u_plane[y2 * srcPitch2 + x2];
	const float V = v_plane[y2 * srcPitch2 + x2];

	const float3 RGB = YUV2RGB(Y, U, V);

	dstImage[y * width + x] = make_vec<T>(RGB.x, RGB.y, RGB.z, 255);
}

template <typename T, bool formatYV12>
static cudaError_t launch420ToRGB(void* srcDev, T* dstDev, size_t width, size_t height, cudaStream_t stream) 
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const int srcPitch = width * sizeof(uint8_t);
	const int dstPitch = width * sizeof(T);

	const dim3 blockDim(8,8);
	//const dim3 gridDim((width+(2*blockDim.x-1))/(2*blockDim.x), (height+(blockDim.y-1))/blockDim.y, 1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height, blockDim.y));

	I420ToRGB<T, formatYV12><<<gridDim, blockDim, 0, stream>>>( (uint8_t*)srcDev, srcPitch, dstDev, dstPitch, width, height );

	return CUDA(cudaGetLastError());
}


// cudaI420ToRGB (uchar3)
cudaError_t cudaI420ToRGB(void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream) 
{
    return launch420ToRGB<uchar3, false>(input, output, width, height, stream);
}

// cudaI420ToRGB (float3)
cudaError_t cudaI420ToRGB(void* input, float3* output, size_t width, size_t height, cudaStream_t stream) 
{
    return launch420ToRGB<float3, false>(input, output, width, height, stream);
}

// cudaI420ToRGBA (uchar4)
cudaError_t cudaI420ToRGBA(void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream) 
{
    return launch420ToRGB<uchar4, false>(input, output, width, height, stream);
}

// cudaI420ToRGBA (float4)
cudaError_t cudaI420ToRGBA(void* input, float4* output, size_t width, size_t height, cudaStream_t stream) 
{
    return launch420ToRGB<float4, false>(input, output, width, height, stream);
}

//-----------------------------------------------------------------------------------

// cudaYV12ToRGB (uchar3)
cudaError_t cudaYV12ToRGB(void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream) 
{
    return launch420ToRGB<uchar3, true>(input, output, width, height, stream);
}

// cudaYV12ToRGB (float3)
cudaError_t cudaYV12ToRGB(void* input, float3* output, size_t width, size_t height, cudaStream_t stream) 
{
    return launch420ToRGB<float3, true>(input, output, width, height, stream);
}

// cudaYV12ToRGBA (uchar4)
cudaError_t cudaYV12ToRGBA(void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream) 
{
    return launch420ToRGB<uchar4, true>(input, output, width, height, stream);
}

// cudaYV12ToRGBA (float4)
cudaError_t cudaYV12ToRGBA(void* input, float4* output, size_t width, size_t height, cudaStream_t stream) 
{
    return launch420ToRGB<float4, true>(input, output, width, height, stream);
}


//-------------------------------------------------------------------------------------
// RGB to I420/YV12
//-------------------------------------------------------------------------------------
inline __device__ void rgb_to_y(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t& y)
{
	y = static_cast<uint8_t>(((int)(30 * r) + (int)(59 * g) + (int)(11 * b)) / 100);
}

inline __device__ void rgb_to_yuv(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t& y, uint8_t& u, uint8_t& v)
{
	rgb_to_y(r, g, b, y);
	u = static_cast<uint8_t>(((int)(-17 * r) - (int)(33 * g) + (int)(50 * b) + 12800) / 100);
	v = static_cast<uint8_t>(((int)(50 * r) - (int)(42 * g) - (int)(8 * b) + 12800) / 100);
}

template <typename T, bool formatYV12>
__global__ void RGBToYV12( T* src, int srcAlignedWidth, uint8_t* dst, int dstPitch, int width, int height )
{
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

	const int x1 = x + 1;
	const int y1 = y + 1;

	if( x1 >= width || y1 >= height )
		return;

	const int planeSize = height * dstPitch;
	
	uint8_t* y_plane = dst;
	uint8_t* u_plane;
	uint8_t* v_plane;

	if( formatYV12 )
	{
		u_plane = y_plane + planeSize;
		v_plane = u_plane + (planeSize / 4);	// size of U & V planes is 25% of Y plane
	}
	else
	{
		v_plane = y_plane + planeSize;		// in I420, order of U & V planes is reversed
		u_plane = v_plane + (planeSize / 4);
	}

	T px;
	uint8_t y_val, u_val, v_val;

	px = src[y * srcAlignedWidth + x];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y * dstPitch + x] = y_val;

	px = src[y * srcAlignedWidth + x1];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y * dstPitch + x1] = y_val;

	px = src[y1 * srcAlignedWidth + x];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y1 * dstPitch + x] = y_val;
	
	px = src[y1 * srcAlignedWidth + x1];
	rgb_to_yuv(px.x, px.y, px.z, y_val, u_val, v_val);
	y_plane[y1 * dstPitch + x1] = y_val;

	const int uvPitch = dstPitch / 2;
	const int uvIndex = (y / 2) * uvPitch + (x / 2);

	u_plane[uvIndex] = u_val;
	v_plane[uvIndex] = v_val;
} 

template<typename T, bool formatYV12>
static cudaError_t launchRGBTo420( T* input, size_t inputPitch, void* output, size_t outputPitch, size_t width, size_t height, cudaStream_t stream)
{
	if( !input || !inputPitch || !output || !outputPitch || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(32, 8);
	const dim3 grid(iDivUp(width, block.x * 2), iDivUp(height, block.y * 2));

	const int inputAlignedWidth = inputPitch / sizeof(T);

	RGBToYV12<T, formatYV12><<<grid, block, 0, stream>>>(input, inputAlignedWidth, (uint8_t*)output, outputPitch, width, height);

	return CUDA(cudaGetLastError());
}


// cudaRGBToI420 (uchar3)
cudaError_t cudaRGBToI420( uchar3* input, size_t inputPitch, void* output, size_t outputPitch, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBTo420<uchar3,true>( input, inputPitch, output, outputPitch, width, height, stream );
}

// cudaRGBToI420 (uchar3)
cudaError_t cudaRGBToI420( uchar3* input, void* output, size_t width, size_t height, cudaStream_t stream )
{
	return cudaRGBToI420( input, width * sizeof(uchar3), output, width * sizeof(uint8_t), width, height, stream );
}

// cudaRGBToI420 (float3)
cudaError_t cudaRGBToI420( float3* input, size_t inputPitch, void* output, size_t outputPitch, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBTo420<float3,true>( input, inputPitch, output, outputPitch, width, height, stream );
}

// cudaRGBAToI420 (float3)
cudaError_t cudaRGBToI420( float3* input, void* output, size_t width, size_t height, cudaStream_t stream )
{
	return cudaRGBToI420( input, width * sizeof(float3), output, width * sizeof(uint8_t), width, height, stream );
}

// cudaRGBAToI420 (uchar4)
cudaError_t cudaRGBAToI420( uchar4* input, size_t inputPitch, void* output, size_t outputPitch, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBTo420<uchar4,true>( input, inputPitch, output, outputPitch, width, height, stream );
}

// cudaRGBAToI420 (uchar4)
cudaError_t cudaRGBAToI420( uchar4* input, void* output, size_t width, size_t height, cudaStream_t stream )
{
	return cudaRGBAToI420( input, width * sizeof(uchar4), output, width * sizeof(uint8_t), width, height, stream );
}

// cudaRGBAToI420 (float4)
cudaError_t cudaRGBAToI420( float4* input, size_t inputPitch, void* output, size_t outputPitch, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBTo420<float4,true>( input, inputPitch, output, outputPitch, width, height, stream );
}

// cudaRGBAToI420 (float4)
cudaError_t cudaRGBAToI420( float4* input, void* output, size_t width, size_t height, cudaStream_t stream )
{
	return cudaRGBAToI420( input, width * sizeof(float4), output, width * sizeof(uint8_t), width, height, stream );
}

//-----------------------------------------------------------------------------------

// cudaRGBToYV12 (uchar3)
cudaError_t cudaRGBToYV12( uchar3* input, size_t inputPitch, void* output, size_t outputPitch, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBTo420<uchar3,false>( input, inputPitch, output, outputPitch, width, height, stream );
}

// cudaRGBToYV12 (uchar3)
cudaError_t cudaRGBToYV12( uchar3* input, void* output, size_t width, size_t height, cudaStream_t stream )
{
	return cudaRGBToYV12( input, width * sizeof(uchar3), output, width * sizeof(uint8_t), width, height, stream );
}

// cudaRGBToYV12 (float3)
cudaError_t cudaRGBToYV12( float3* input, size_t inputPitch, void* output, size_t outputPitch, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBTo420<float3,false>( input, inputPitch, output, outputPitch, width, height, stream );
}

// cudaRGBToYV12 (float3)
cudaError_t cudaRGBToYV12( float3* input, void* output, size_t width, size_t height, cudaStream_t stream )
{
	return cudaRGBToYV12( input, width * sizeof(float3), output, width * sizeof(uint8_t), width, height, stream );
}

// cudaRGBAToYV12 (uchar4)
cudaError_t cudaRGBAToYV12( uchar4* input, size_t inputPitch, void* output, size_t outputPitch, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBTo420<uchar4,false>( input, inputPitch, output, outputPitch, width, height, stream );
}

// cudaRGBAToYV12 (uchar4)
cudaError_t cudaRGBAToYV12( uchar4* input, void* output, size_t width, size_t height, cudaStream_t stream )
{
	return cudaRGBAToYV12( input, width * sizeof(uchar4), output, width * sizeof(uint8_t), width, height, stream );
}

// cudaRGBAToYV12 (float4)
cudaError_t cudaRGBAToYV12( float4* input, size_t inputPitch, void* output, size_t outputPitch, size_t width, size_t height, cudaStream_t stream )
{
	return launchRGBTo420<float4,false>( input, inputPitch, output, outputPitch, width, height, stream );
}

// cudaRGBAToYV12 (float4)
cudaError_t cudaRGBAToYV12( float4* input, void* output, size_t width, size_t height, cudaStream_t stream )
{
	return cudaRGBAToYV12( input, width * sizeof(float4), output, width * sizeof(uint8_t), width, height, stream );
}


