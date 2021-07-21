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


#define MIN(a,b)  (a < b ? a : b)
#define MAX(a,b)  (a > b ? a : b)


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
cudaError_t cudaDrawCircle( void* input, void* output, size_t width, size_t height, imageFormat format, int cx, int cy, float radius, const float4& color )
{
	if( !input || !output || width == 0 || height == 0 || radius <= 0 )
		return cudaErrorInvalidValue;

	// if the input and output images are different, copy the input to the output
	// this is because we only launch the kernel in the approximate area of the circle
	if( input != output )
		CUDA(cudaMemcpy(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice));
		
	// find a box around the circle
	const int diameter = ceilf(radius * 2.0f);
	const int offset_x = cx - radius;
	const int offset_y = cy - radius;
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(diameter,blockDim.x), iDivUp(diameter,blockDim.y));

	#define LAUNCH_DRAW_CIRCLE(type) \
		gpuDrawCircle<type><<<gridDim, blockDim>>>((type*)output, width, height, offset_x, offset_y, cx, cy, radius*radius, color)
	
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
// 

#if 0
function sqr(x) { return x * x }
function dist2(v, w) { return sqr(v.x - w.x) + sqr(v.y - w.y) }

template<typename T>
inline __device__ float lineDistanceSquared(T x, T y, T x1, T y1, T x2, T y2) 
{
	const float dist2 = (x2 - x1) * (x2 - x1) + (y2 - y1) + (y2 - y1);
	
function distToSegmentSquared(p, v, w) {
  var l2 = dist2(v, w);
  if (l2 == 0) return dist2(p, v);
  var t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
  t = Math.max(0, Math.min(1, t));
  return dist2(p, { x: v.x + t * (w.x - v.x),
                    y: v.y + t * (w.y - v.y) });
}
#else

template<typename T>
inline __device__ float lineDistanceSquared(T x, T y, T x1, T y1, T x2, T y2) 
{
	const T A = x - x1;
	const T B = y - y1;
	const T C = x2 - x1;
	const T D = y2 - y1;

	const float dot = A * C + B * D;
	const float len_sq = C * C + D * D;
	
	float param = -1;
	
	if (len_sq != 0) //in case of 0 length line
		param = dot / len_sq;

	T xx, yy;

	if (param < 0) {
		xx = x1;
		yy = y1;
	}
	else if (param > 1) {
		xx = x2;
		yy = y2;
	}
	else {
		xx = x1 + param * C;
		yy = y1 + param * D;
	}

	T dx = x - xx;
	T dy = y - yy;
	
	return dx * dx + dy * dy;
}

#endif
				 
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
cudaError_t cudaDrawLine( void* input, void* output, size_t width, size_t height, imageFormat format, int x1, int y1, int x2, int y2, const float4& color, float line_width )
{
	if( !input || !output || width == 0 || height == 0 || line_width <= 0 )
		return cudaErrorInvalidValue;

	// if the input and output images are different, copy the input to the output
	// this is because we only launch the kernel in the approximate area of the circle
	if( input != output )
		CUDA(cudaMemcpy(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice));
		
	// find a box around the line
	const int left = MIN(x1,x2);
	const int right = MAX(x1,x2);
	const int top = MIN(y1,y2);
	const int bottom = MAX(y1,y2);

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(right - left, blockDim.x), iDivUp(bottom - top, blockDim.y));

	#define LAUNCH_DRAW_LINE(type) \
		gpuDrawLine<type><<<gridDim, blockDim>>>((type*)output, width, height, left, top, x1, y1, x2, y2, color, line_width * line_width)
	
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