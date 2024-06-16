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

#ifndef __CUDA_OVERLAY_H__
#define __CUDA_OVERLAY_H__


#include "cudaUtility.h"
#include "imageFormat.h"


/**
 * Overlay the input image onto the output image at location (x,y)
 * If the composted image doesn't entirely fit in the output, it will be cropped.
 * If the images have an alpha channel, they will be alpha blended. 
 * @ingroup overlay
 */
cudaError_t cudaOverlay( void* input, size_t inputWidth, size_t inputHeight,
                         void* output, size_t outputWidth, size_t outputHeight,
                         imageFormat format, int x, int y, cudaStream_t stream=0 );
			
/**
 * Overlay the input image composted onto the output image at location (x,y)
 * If the composted image doesn't entirely fit in the output, it will be cropped.
 * If the images have an alpha channel, they will be alpha blended.
 * @ingroup overlay
 */
template<typename T> 
cudaError_t cudaOverlay( T* input, size_t inputWidth, size_t inputHeight,
                         T* output, size_t outputWidth, size_t outputHeight,
                         int x, int y, cudaStream_t stream=0 )
{ 
	return cudaOverlay(input, inputWidth, inputHeight, output, outputWidth, outputHeight, imageFormatFromType<T>(), x, y, stream); 
}

/**
 * Overlay the input image composted onto the output image at location (x,y)
 * If the composted image doesn't entirely fit in the output, it will be cropped.
 * If the images have an alpha channel, they will be alpha blended.
 * @ingroup overlay
 */
template<typename T> 
cudaError_t cudaOverlay( T* input, const int2& inputDims,
                         T* output, const int2& outputDims,
                         int x, int y, cudaStream_t stream=0 )
{ 
	return cudaOverlay(input, inputDims.x, inputDims.y, output, outputDims.x, outputDims.y, imageFormatFromType<T>(), x, y, stream); 
}
		
		
/**
 * cudaRectFill
 * @deprecated please use the functions from cudaDraw.h instead
 * @ingroup overlay
 */
cudaError_t cudaRectFill( void* input, void* output, size_t width, size_t height, imageFormat format, 
						  float4* rects, int numRects, const float4& color, cudaStream_t stream=0 );

/**
 * cudaRectFill
 * @deprecated please use the functions from cudaDraw.h instead
 * @ingroup overlay
 */
template<typename T> 
cudaError_t cudaRectFill( T* input, T* output, size_t width, size_t height, 
				 		  float4* rects, int numRects, const float4& color,
				 		  cudaStream_t stream=0 )	
{ 
	return cudaRectFill(input, output, width, height, imageFormatFromType<T>(), rects, numRects, color, stream); 
}


#endif
