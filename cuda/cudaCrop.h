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
#include "imageFormat.h"


/**
 * Crop a uint8 grayscale image to the specified region of interest (ROI).
 *
 * @param[in] input Pointer to the input image in CUDA memory.
 * @param[out] output Pointer to the output image in CUDA memory.
 *                    The output image should have the same dimensions as the ROI.
 * @param roi The region of interest from the input image that will be copied to
 *            the output image. This `int4` vector forms a crop rectangle as follows:
 *
 *               - `roi.x -> left`
 *               - `roi.y -> top`
 *               - `roi.z -> right`
 *               - `roi.w -> bottom`
 *
 *            The ROI coordinates must not be negative, otherwise an error will be returned.
 *
 * @param inputWidth width of the input image (in pixels)
 * @param inputWidth height of the input image (in pixels)
 *
 * @ingroup crop
 */
cudaError_t cudaCrop( uint8_t* input, uint8_t* output, const int4& roi, size_t inputWidth, size_t inputHeight );

/**
 * Crop a floating-point grayscale image to the specified region of interest (ROI).
 *
 * @param[in] input Pointer to the input image in CUDA memory.
 * @param[out] output Pointer to the output image in CUDA memory.
 *                    The output image should have the same dimensions as the ROI.
 * @param roi The region of interest from the input image that will be copied to
 *            the output image. This `int4` vector forms a crop rectangle as follows:
 *
 *               - `roi.x -> left`
 *               - `roi.y -> top`
 *               - `roi.z -> right`
 *               - `roi.w -> bottom`
 *
 *            The ROI coordinates must not be negative, otherwise an error will be returned.
 *
 * @param inputWidth width of the input image (in pixels)
 * @param inputWidth height of the input image (in pixels)
 *
 * @ingroup crop
 */
cudaError_t cudaCrop( float* input, float* output, const int4& roi, size_t inputWidth, size_t inputHeight );

/**
 * Crop a uchar3 RGB/BGR image to the specified region of interest (ROI).
 *
 * @param[in] input Pointer to the input image in CUDA memory.
 * @param[out] output Pointer to the output image in CUDA memory.
 *                    The output image should have the same dimensions as the ROI.
 * @param roi The region of interest from the input image that will be copied to
 *            the output image. This `int4` vector forms a crop rectangle as follows:
 *
 *               - `roi.x -> left`
 *               - `roi.y -> top`
 *               - `roi.z -> right`
 *               - `roi.w -> bottom`
 *
 *            The ROI coordinates must not be negative, otherwise an error will be returned.
 *
 * @param inputWidth width of the input image (in pixels)
 * @param inputWidth height of the input image (in pixels)
 *
 * @ingroup crop
 */
cudaError_t cudaCrop( uchar3* input, uchar3* output, const int4& roi, size_t inputWidth, size_t inputHeight );

/**
 * Crop a uchar4 RGBA/BGRA image to the specified region of interest (ROI).
 *
 * @param[in] input Pointer to the input image in CUDA memory.
 * @param[out] output Pointer to the output image in CUDA memory.
 *                    The output image should have the same dimensions as the ROI.
 * @param roi The region of interest from the input image that will be copied to
 *            the output image. This `int4` vector forms a crop rectangle as follows:
 *
 *               - `roi.x -> left`
 *               - `roi.y -> top`
 *               - `roi.z -> right`
 *               - `roi.w -> bottom`
 *
 *            The ROI coordinates must not be negative, otherwise an error will be returned.
 *
 * @param inputWidth width of the input image (in pixels)
 * @param inputWidth height of the input image (in pixels)
 *
 * @ingroup crop
 */
cudaError_t cudaCrop( uchar4* input, uchar4* output, const int4& roi, size_t inputWidth, size_t inputHeight );

/**
 * Crop a float3 RGB/BGR image to the specified region of interest (ROI).
 *
 * @param[in] input Pointer to the input image in CUDA memory.
 * @param[out] output Pointer to the output image in CUDA memory.
 *                    The output image should have the same dimensions as the ROI.
 * @param roi The region of interest from the input image that will be copied to
 *            the output image. This `int4` vector forms a crop rectangle as follows:
 *
 *               - `roi.x -> left`
 *               - `roi.y -> top`
 *               - `roi.z -> right`
 *               - `roi.w -> bottom`
 *
 *            The ROI coordinates must not be negative, otherwise an error will be returned.
 *
 * @param inputWidth width of the input image (in pixels)
 * @param inputWidth height of the input image (in pixels)
 *
 * @ingroup crop
 */
cudaError_t cudaCrop( float3* input, float3* output, const int4& roi, size_t inputWidth, size_t inputHeight );

/**
 * Crop a float4 RGBA/BGRA image to the specified region of interest (ROI).
 *
 * @param[in] input Pointer to the input image in CUDA memory.
 * @param[out] output Pointer to the output image in CUDA memory.
 *                    The output image should have the same dimensions as the ROI.
 * @param roi The region of interest from the input image that will be copied to
 *            the output image. This `int4` vector forms a crop rectangle as follows:
 *
 *               - `roi.x -> left`
 *               - `roi.y -> top`
 *               - `roi.z -> right`
 *               - `roi.w -> bottom`
 *
 *            The ROI coordinates must not be negative, otherwise an error will be returned.
 *
 * @param inputWidth width of the input image (in pixels)
 * @param inputWidth height of the input image (in pixels)
 *
 * @ingroup crop
 */
cudaError_t cudaCrop( float4* input, float4* output, const int4& roi, size_t inputWidth, size_t inputHeight );

/**
 * Crop a float4 RGBA/BGRA image to the specified region of interest (ROI).
 *
 * @param[in] input Pointer to the input image in CUDA memory.
 * @param[out] output Pointer to the output image in CUDA memory.
 *                    The output image should have the same dimensions as the ROI.
 * @param roi The region of interest from the input image that will be copied to
 *            the output image. This `int4` vector forms a crop rectangle as follows:
 *
 *               - `roi.x -> left`
 *               - `roi.y -> top`
 *               - `roi.z -> right`
 *               - `roi.w -> bottom`
 *
 *            The ROI coordinates must not be negative, otherwise an error will be returned.
 *
 * @param inputWidth width of the input image (in pixels)
 * @param inputWidth height of the input image (in pixels)
 * @param format format of the image - valid formats are gray8, gray32f, rgb8/bgr8, 
 *               rgba8/bgra8, rgb32f/bgr32f, and rgba32f/bgra32f.
 *
 * @ingroup crop
 */
cudaError_t cudaCrop( void* input, void* output, const int4& roi, size_t inputWidth, size_t inputHeight, imageFormat format );


#endif

