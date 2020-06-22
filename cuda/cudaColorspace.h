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

#ifndef __CUDA_COLORSPACE_H__
#define __CUDA_COLORSPACE_H__

#include "cudaUtility.h"
#include "imageFormat.h"


/**
 * Convert between two image formats using the GPU.
 *
 * This function supports various conversions between RGB/RGBA, BGR/BGRA, Bayer, grayscale, and YUV.
 * In addition to converting between the previously listed colorspaces, you can also change the 
 * number of channels or data format (for example, `IMAGE_RGB8` to `IMAGE_RGBA32F`).
 *
 * For the list of image formats available, @see imageFormat enum.
 *
 * Limitations and unsupported conversions include:
 *
 *     - The YUV formats don't support BGR/BGRA or grayscale (RGB/RGBA only)
 *     - YUV NV12, YUYV, YVYU, and UYVY can only be converted to RGB/RGBA (not from)
 *     - Bayer formats can only be converted to RGB8 (`uchar3`) and RGBA8 (`uchar4`)
 *
 * @param input CUDA device pointer to the input image
 * @param inputFormat format enum of the input image
 * @param output CUDA device pointer to the input image
 * @param outputFormat format enum of the output image
 * @param width width of the input and output images (in pixels)
 * @param height height of the input and output images (in pixels)
 * @param pixel_range for floating-point to 8-bit conversions, specifies the range of pixel intensities
 *                    in the input image that get normalized to `[0,255]`.  The default input range is
 *                    `[0,255]`, and as such no normalization occurs.  Other common pixel ranges include
 *                    `[0,1]` and `[-1,1]`, and these pixel values would be re-scaled for `[0,255]` output.
 *                    Note that this parameter is only used for float-to-uchar conversions where the data
 *                    is downcast (for example, `IMAGE_RGB32F` to `IMAGE_RGB8`).
 * @ingroup colorspace
 */
cudaError_t cudaConvertColor( void* input, imageFormat inputFormat,
					     void* output, imageFormat outputFormat,
					     size_t width, size_t height,
						const float2& pixel_range=make_float2(0,255));

/**
 * Convert between to image formats using the GPU.
 *
 * This templated overload of cudaConvertColor() supports uchar3 (`IMAGE_RGB8`), 
 * uchar4 (`IMAGE_RGBA8`), float3 (`IMAGE_RGB32F`), and float4 (`IMAGE_RGBA32F`).
 *
 * To convert to/from other formats such as YUV, grayscale, and BGR/BGRA, see 
 * the other version of cudaConvertColor() that uses explicity imageFormat enums.
 *
 * @param input CUDA device pointer to the input image
 * @param output CUDA device pointer to the input image
 * @param width width of the input and output images (in pixels)
 * @param height height of the input and output images (in pixels)
 * @param pixel_range for floating-point to 8-bit conversions, specifies the range of pixel intensities
 *                    in the input image that get normalized to `[0,255]`.  The default input range is
 *                    `[0,255]`, and as such no normalization occurs.  Other common pixel ranges include
 *                    `[0,1]` and `[-1,1]`, and these pixel values would be re-scaled for `[0,255]` output.
 *                    Note that this parameter is only used for float-to-uchar conversions where the data
 *                    is downcast (for example, `IMAGE_RGB32F` to `IMAGE_RGB8`).
 *
 * @ingroup colorspace
 */
template<typename T_in, typename T_out> 
cudaError_t cudaConvertColor( T_in* input, T_out* output,
					     size_t width, size_t height,
						const float2& pixel_range=make_float2(0,255))	
{ 
	return cudaConvertColor(input, imageFormatFromType<T_in>(), output, imageFormatFromType<T_out>(), width, height, pixel_range); 
}
	

#endif

