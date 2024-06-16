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

#ifndef __CUDA_DRAW_H__
#define __CUDA_DRAW_H__


#include "cudaUtility.h"
#include "imageFormat.h"


/**
 * cudaDrawCircle
 * @ingroup drawing
 */
cudaError_t cudaDrawCircle( void* input, void* output, size_t width, size_t height, imageFormat format, 
                            int cx, int cy, float radius, const float4& color, cudaStream_t stream=0 );
	
/**
 * cudaDrawCircle
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawCircle( T* input, T* output, size_t width, size_t height, 
                            int cx, int cy, float radius, const float4& color,
                            cudaStream_t stream=0 )	
{ 
	return cudaDrawCircle(input, output, width, height, imageFormatFromType<T>(), cx, cy, radius, color, stream); 
}	

/**
 * cudaDrawCircle (in-place)
 * @ingroup drawing
 */
inline cudaError_t cudaDrawCircle( void* image, size_t width, size_t height, imageFormat format, 
                                   int cx, int cy, float radius, const float4& color, cudaStream_t stream=0 )
{
	return cudaDrawCircle(image, image, width, height, format, cx, cy, radius, color, stream);
}

/**
 * cudaDrawCircle (in-place)
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawCircle( T* image, size_t width, size_t height, 
                            int cx, int cy, float radius, const float4& color,
                            cudaStream_t stream=0 )	
{ 
	return cudaDrawCircle(image, width, height, imageFormatFromType<T>(), cx, cy, radius, color, stream); 
}

/**
 * cudaDrawLine
 * @ingroup drawing
 */
cudaError_t cudaDrawLine( void* input, void* output, size_t width, size_t height, imageFormat format, 
                          int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0,
                          cudaStream_t stream=0 );
	
/**
 * cudaDrawLine
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawLine( T* input, T* output, size_t width, size_t height, 
                          int x1, int y1, int x2, int y2, const float4& color, 
                          float line_width=1.0, cudaStream_t stream=0 )	
{ 
	return cudaDrawLine(input, output, width, height, imageFormatFromType<T>(), x1, y1, x2, y2, color, line_width, stream); 
}

/**
 * cudaDrawLine (in-place)
 * @ingroup drawing
 */
inline cudaError_t cudaDrawLine( void* image, size_t width, size_t height, imageFormat format, 
                                 int x1, int y1, int x2, int y2, const float4& color, 
                                 float line_width=1.0, cudaStream_t stream=0 )
{
	return cudaDrawLine(image, image, width, height, format, x1, y1, x2, y2, color, line_width, stream);
}					
	
/**
 * cudaDrawLine (in-place)
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawLine( T* image, size_t width, size_t height, 
                          int x1, int y1, int x2, int y2, const float4& color, 
                          float line_width=1.0, cudaStream_t stream=0 )	
{ 
	return cudaDrawLine(image, width, height, imageFormatFromType<T>(), x1, y1, x2, y2, color, line_width, stream); 
}	


/**
 * cudaDrawRect
 * @ingroup drawing
 */
cudaError_t cudaDrawRect( void* input, void* output, size_t width, size_t height, imageFormat format, 
                          int left, int top, int right, int bottom, const float4& color, 
                          const float4& line_color=make_float4(0,0,0,0), float line_width=1.0f,
                          cudaStream_t stream=0 );

/**
 * cudaDrawRect
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawRect( T* input, T* output, size_t width, size_t height, 
                          int left, int top, int right, int bottom, const float4& color,
                          const float4& line_color=make_float4(0,0,0,0), float line_width=1.0f,
                          cudaStream_t stream=0 )	
{ 
	return cudaDrawRect(input, output, width, height, imageFormatFromType<T>(), left, top, right, bottom, color, line_color, line_width, stream); 
}

/**
 * cudaDrawRect (in-place)
 * @ingroup drawing
 */
inline cudaError_t cudaDrawRect( void* image, size_t width, size_t height, imageFormat format, 
                                 int left, int top, int right, int bottom, const float4& color,
                                 const float4& line_color=make_float4(0,0,0,0), float line_width=1.0f,
                                 cudaStream_t stream=0 )
{
	return cudaDrawRect(image, image, width, height, format, left, top, right, bottom, color, line_color, line_width, stream);
}

/**
 * cudaDrawRect
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawRect( T* image, size_t width, size_t height, 
                          int left, int top, int right, int bottom, const float4& color,
                          const float4& line_color=make_float4(0,0,0,0), float line_width=1.0f,
                          cudaStream_t stream=0 )	
{ 
	return cudaDrawRect(image, image, width, height, imageFormatFromType<T>(), left, top, right, bottom, color, line_color, line_width, stream); 
}

#endif
