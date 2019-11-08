/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __CUDA_CROP_H__
#define __CUDA_CROP_H__


#include "cudaUtility.h"


/**
 * Crop a single-channel grayscale image to the specified region of interest (ROI).
 *
 * @param[in] input Pointer to the single-channel float input image in CUDA memory.
 * @param[out] output Pointer to the single-channel float output image in CUDA memory.
 *                    The output image should have the same dimensions as the ROI.
 * @param roi The region of interest from the input image that will be copied to
 *            the output image. This `int4` vector forms a crop rectangle as follows:
 *
 *               - `roi.x -> left`
 *               - `roi.y -> top`
 *               - `roi.z -> right`
 *               - `roi.w -> bottom`
 *
 * @param inputWidth width of the input image (in pixels)
 * @param inputWidth height of the input image (in pixels)
 *
 * @ingroup cuda
 */
cudaError_t cudaCrop( float* input, float* output, const int4& roi, 
				  size_t inputWidth, size_t inputHeight );


/**
 * Crop an RGBA image to the specified region of interest (ROI).
 *
 * @param[in] input Pointer to the float4 RGBA input image in CUDA memory.
 * @param[out] output Pointer to the float4 RGBA output image in CUDA memory.
 *                    The output image should have the same dimensions as the ROI.
 * @param roi The region of interest from the input image that will be copied to
 *            the output image. This `int4` vector forms a crop rectangle as follows:
 *
 *               - `roi.x -> left`
 *               - `roi.y -> top`
 *               - `roi.z -> right`
 *               - `roi.w -> bottom`
 *
 * @param inputWidth width of the input image (in pixels)
 * @param inputWidth height of the input image (in pixels)
 *
 * @ingroup cuda
 */
cudaError_t cudaCropRGBA( float4* input, float4* output, const int4& roi, 
					 size_t inputWidth, size_t inputHeight );

#endif

