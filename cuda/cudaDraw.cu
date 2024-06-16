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

#include "cudaDraw.h"
#include "cudaAlphaBlend.cuh"


// TODO for rect/fill/line
//    - make versions that only accept image (as both input/output)
//    - add line width/line color
//    - add overloads for single shape/multiple shapes
//    - benchmarking of copy vs alternate kernel when input != output
//    - overloads using int2 for coordinates
//    - add a template parameter for alpha blending

#define MIN(a,b)  (a < b ? a : b)
#define MAX(a,b)  (a > b ? a : b)

template<typename T> inline __device__ __host__ T sqr(T x) 				    { return x*x; }

inline __device__ __host__ float dist2(float x1, float y1, float x2, float y2) { return sqr(x1-x2) + sqr(y1-y2); }
inline __device__ __host__ float dist(float x1, float y1, float x2, float y2)  { return sqrtf(dist2(x1,y1,x2,y2)); }


//----------------------------------------------------------------------------
// Circle drawing (find if the distance to the circle <= radius)
//----------------------------------------------------------------------------						 
template<typename T>
__global__ void gpuDrawCircle( T* img, int imgWidth, int imgHeight, int offset_x, int offset_y, int cx, int cy, float radius2, const float4 color ) 
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x + offset_x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y + offset_y;

	if( x >= imgWidth || y >= imgHeight || x < 0 || y < 0 )
		return;

	const int dx = x - cx;
	const int dy = y - cy;
	
	// if x,y is in the circle draw it
	if( dx * dx + dy * dy < radius2 ) 
	{
		const int idx = y * imgWidth + x;
		img[idx] = cudaAlphaBlend(img[idx], color);
	}
}

// cudaDrawCircle
cudaError_t cudaDrawCircle( void* input, void* output, size_t width, size_t height, imageFormat format, int cx, int cy, float radius, const float4& color, cudaStream_t stream )
{
	if( !input || !output || width == 0 || height == 0 || radius <= 0 )
		return cudaErrorInvalidValue;

	// if the input and output images are different, copy the input to the output
	// this is because we only launch the kernel in the approximate area of the circle
	if( input != output )
		CUDA(cudaMemcpyAsync(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice, stream));
		
	// find a box around the circle
	const int diameter = ceilf(radius * 2.0f);
	const int offset_x = cx - radius;
	const int offset_y = cy - radius;
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(diameter,blockDim.x), iDivUp(diameter,blockDim.y));

	#define LAUNCH_DRAW_CIRCLE(type) \
		gpuDrawCircle<type><<<gridDim, blockDim, 0, stream>>>((type*)output, width, height, offset_x, offset_y, cx, cy, radius*radius, color)
	
	if( format == IMAGE_RGB8 )
		LAUNCH_DRAW_CIRCLE(uchar3);
	else if( format == IMAGE_RGBA8 )
		LAUNCH_DRAW_CIRCLE(uchar4);
	else if( format == IMAGE_RGB32F )
		LAUNCH_DRAW_CIRCLE(float3); 
	else if( format == IMAGE_RGBA32F )
		LAUNCH_DRAW_CIRCLE(float4);
	else
	{
		imageFormatErrorMsg(LOG_CUDA, "cudaDrawCircle()", format);
		return cudaErrorInvalidValue;
	}
		
	return cudaGetLastError();
}


//----------------------------------------------------------------------------
// Line drawing (find if the distance to the line <= line_width)
// Distance from point to line segment - https://stackoverflow.com/a/1501725
//----------------------------------------------------------------------------
inline __device__ float lineDistanceSquared(float x, float y, float x1, float y1, float x2, float y2) 
{
	const float d = dist2(x1, y1, x2, y2);
	const float t = ((x-x1) * (x2-x1) + (y-y1) * (y2-y1)) / d;
	const float u = MAX(0, MIN(1, t));
	
	return dist2(x, y, x1 + u * (x2 - x1), y1 + u * (y2 - y1));
}
				 
template<typename T>
__global__ void gpuDrawLine( T* img, int imgWidth, int imgHeight, int offset_x, int offset_y, int x1, int y1, int x2, int y2, const float4 color, float line_width2 ) 
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x + offset_x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y + offset_y;

	if( x >= imgWidth || y >= imgHeight || x < 0 || y < 0 )
		return;

	if( lineDistanceSquared(x, y, x1, y1, x2, y2) <= line_width2 )
	{
		const int idx = y * imgWidth + x;
		img[idx] = cudaAlphaBlend(img[idx], color);
	}
}

// cudaDrawLine
cudaError_t cudaDrawLine( void* input, void* output, size_t width, size_t height, imageFormat format, int x1, int y1, int x2, int y2, const float4& color, float line_width, cudaStream_t stream )
{
	if( !input || !output || width == 0 || height == 0 || line_width <= 0 )
		return cudaErrorInvalidValue;
	
	// check for lines < 2 pixels in length
	if( dist(x1,y1,x2,y2) < 2.0 )
	{
		LogWarning(LOG_CUDA "cudaDrawLine() - line has length < 2, skipping (%i,%i) (%i,%i)\n", x1, y1, x2, y2);
		return cudaSuccess;
	}
	
	// if the input and output images are different, copy the input to the output
	// this is because we only launch the kernel in the approximate area of the circle
	if( input != output )
		CUDA(cudaMemcpyAsync(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice, stream));
		
	// find a box around the line
	const int left = MIN(x1,x2) - line_width;
	const int right = MAX(x1,x2) + line_width;
	const int top = MIN(y1,y2) - line_width;
	const int bottom = MAX(y1,y2) + line_width;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(right - left, blockDim.x), iDivUp(bottom - top, blockDim.y));

	#define LAUNCH_DRAW_LINE(type) \
		gpuDrawLine<type><<<gridDim, blockDim, 0, stream>>>((type*)output, width, height, left, top, x1, y1, x2, y2, color, line_width * line_width)
	
	if( format == IMAGE_RGB8 )
		LAUNCH_DRAW_LINE(uchar3);
	else if( format == IMAGE_RGBA8 )
		LAUNCH_DRAW_LINE(uchar4);
	else if( format == IMAGE_RGB32F )
		LAUNCH_DRAW_LINE(float3); 
	else if( format == IMAGE_RGBA32F )
		LAUNCH_DRAW_LINE(float4);
	else
	{
		imageFormatErrorMsg(LOG_CUDA, "cudaDrawLine()", format);
		return cudaErrorInvalidValue;
	}
		
	return cudaGetLastError();
}



//----------------------------------------------------------------------------
// Rect drawing (a grid of threads is launched over the rect)
//----------------------------------------------------------------------------
template<typename T>
__global__ void gpuDrawRect( T* img, int imgWidth, int imgHeight, int x0, int y0, int boxWidth, int boxHeight, const float4 color ) 
{
	const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	if( box_x >= boxWidth || box_y >= boxHeight )
		return;

	const int x = box_x + x0;
	const int y = box_y + y0;

	if( x >= imgWidth || y >= imgHeight || x < 0 || y < 0 )
		return;

	const int idx = y * imgWidth + x;
	img[idx] = cudaAlphaBlend(img[idx], color);
}


// cudaDrawRect
cudaError_t cudaDrawRect( void* input, void* output, size_t width, size_t height, imageFormat format, int left, int top, int right, int bottom, const float4& color, const float4& line_color, float line_width, cudaStream_t stream )
{
	if( !input || !output || width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// if the input and output images are different, copy the input to the output
	// this is because we only launch the kernel in the approximate area of the circle
	if( input != output )
		CUDA(cudaMemcpyAsync(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice, stream));
		
	// make sure the coordinates are ordered
	if( left > right )
	{
		const int swap = left;
		left = right;
		right = swap;
	}
	
	if( top > bottom )
	{
		const int swap = top;
		top = bottom;
		bottom = swap;
	}
	
	const int boxWidth = right - left;
	const int boxHeight = bottom - top;
	
	if( boxWidth <= 0 || boxHeight <= 0 )
	{
		LogError(LOG_CUDA "cudaDrawRect() -- rect had width/height <= 0  left=%i top=%i right=%i bottom=%i\n", left, top, right, bottom);
		return cudaErrorInvalidValue;
	}

	// rect fill
	if( color.w > 0 )
	{
		const dim3 blockDim(8, 8);
		const dim3 gridDim(iDivUp(boxWidth,blockDim.x), iDivUp(boxHeight,blockDim.y));
				
		#define LAUNCH_DRAW_RECT(type) \
			gpuDrawRect<type><<<gridDim, blockDim, 0, stream>>>((type*)output, width, height, left, top, boxWidth, boxHeight, color)
		
		if( format == IMAGE_RGB8 )
			LAUNCH_DRAW_RECT(uchar3);
		else if( format == IMAGE_RGBA8 )
			LAUNCH_DRAW_RECT(uchar4);
		else if( format == IMAGE_RGB32F )
			LAUNCH_DRAW_RECT(float3); 
		else if( format == IMAGE_RGBA32F )
			LAUNCH_DRAW_RECT(float4);
		else
		{
			imageFormatErrorMsg(LOG_CUDA, "cudaDrawRect()", format);
			return cudaErrorInvalidValue;
		}
	}
	
	// rect outline
	if( line_color.w > 0 && line_width > 0 )
	{
		int lines[4][4] = {
			{left, top, right, top},
			{right, top, right, bottom},
			{right, bottom, left, bottom},
			{left, bottom, left, top}
		};
		
		for( uint32_t n=0; n < 4; n++ )
			CUDA(cudaDrawLine(output, width, height, format, lines[n][0], lines[n][1], lines[n][2], lines[n][3], line_color, line_width, stream));
	}
	
	return cudaGetLastError();
}
