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

#ifndef __CUDA_WARP_H__
#define __CUDA_WARP_H__


#include "cudaUtility.h"
#include "imageFormat.h"


/**
 * Apply 2x3 affine warp to an 8-bit fixed-point RGBA image.
 * The 2x3 matrix transform is in row-major order (transform[row][column])
 * If the transform has already been inverted, set transform_inverted to true.
 * @ingroup warping
 */
cudaError_t cudaWarpAffine( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
                            const float transform[2][3], bool transform_inverted=false,
                            cudaStream_t stream=0 );


/**
 * Apply 2x3 affine warp to an 32-bit floating-point RGBA image.
 * The 2x3 matrix transform is in row-major order (transform[row][column])
 * If the transform has already been inverted, set transform_inverted to true.
 * @ingroup warping
 */
cudaError_t cudaWarpAffine( float4* input, float4* output, uint32_t width, uint32_t height,
                            const float transform[2][3], bool transform_inverted=false,
                            cudaStream_t stream=0 );


/**
 * Apply a 3x3 perspective warp to an image.
 * The 3x3 matrix transform is in row-major order (transform[row][column])
 * If the transform has already been inverted, set transform_inverted to true.
 * @ingroup warping
 */
cudaError_t cudaWarpPerspective( void* input, uint32_t inputWidth, uint32_t inputHeight, imageFormat inputFormat,
                                 void* output, uint32_t outputWidth, uint32_t outputHeight, imageFormat outputFormat,
                                 const float transform[3][3], bool transform_inverted=false, cudaStream_t stream=0 );
		
		
/**
 * Apply a 3x3 perspective warp to an image.
 * The 3x3 matrix transform is in row-major order (transform[row][column])
 * If the transform has already been inverted, set transform_inverted to true.
 * @ingroup warping
 */
template<typename T> 
cudaError_t cudaWarpPerspective( T* input, uint32_t inputWidth, uint32_t inputHeight,
                                 T* output, uint32_t outputWidth, uint32_t outputHeight,
                                 const float transform[3][3], bool transform_inverted=false,
                                 cudaStream_t stream=0 )
{ 
	return cudaWarpPerspective(input, inputWidth, inputHeight, imageFormatFromType<T>(), output, outputWidth, outputHeight, imageFormatFromType<T>(), transform, transform_inverted, stream);
}	

						   
/**
 * Apply 3x3 perspective warp to an 8-bit fixed-point RGBA image.
 * The 3x3 matrix transform is in row-major order (transform[row][column])
 * If the transform has already been inverted, set transform_inverted to true.
 * @deprecated this function has been deprecated in lieu of the overload that
 *             accepts different dimensions for the input and output images.
 * @ingroup warping
 */
cudaError_t cudaWarpPerspective( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
                                 const float transform[3][3], bool transform_inverted=false,
                                 cudaStream_t stream=0 );


/**
 * Apply 3x3 perspective warp to an 32-bit floating-point RGBA image.
 * The 3x3 matrix transform is in row-major order (transform[row][column])
 * If the transform has already been inverted, set transform_inverted to true.
 * @deprecated this function has been deprecated in lieu of the overload that
 *             accepts different dimensions for the input and output images.
 * @ingroup warping
 */
cudaError_t cudaWarpPerspective( float4* input, float4* output, uint32_t width, uint32_t height,
                                 const float transform[3][3], bool transform_inverted=false,
                                 cudaStream_t stream=0 );


/**
 * Apply in-place instrinsic lens distortion correction to an 8-bit fixed-point RGBA image.
 * Pinhole camera model with radial (barrel) distortion and tangential distortion.
 * @ingroup warping
 */
cudaError_t cudaWarpIntrinsic( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
                               const float2& focalLength, const float2& principalPoint, const float4& distortion,
                               cudaStream_t stream=0 );
											  

/**
 * Apply in-place instrinsic lens distortion correction to 32-bit floating-point RGBA image.
 * Pinhole camera model with radial (barrel) distortion and tangential distortion.
 * @ingroup warping
 */
cudaError_t cudaWarpIntrinsic( float4* input, float4* output, uint32_t width, uint32_t height,
                               const float2& focalLength, const float2& principalPoint, const float4& distortion,
                               cudaStream_t stream=0 );
											  

/**
 * Apply fisheye lens dewarping to an 8-bit fixed-point RGBA image.
 * @param[in] focus focus of the lens (in mm).
 * @ingroup warping
 */
cudaError_t cudaWarpFisheye( uchar4* input, uchar4* output, uint32_t width, uint32_t height, float focus, cudaStream_t stream=0 );


/**
 * Apply fisheye lens dewarping to a 32-bit floating-point RGBA image.
 * @param[in] focus focus of the lens (in mm).
 * @ingroup warping
 */
cudaError_t cudaWarpFisheye( float4* input, float4* output, uint32_t width, uint32_t height, float focus, cudaStream_t stream=0 );

							
#endif

