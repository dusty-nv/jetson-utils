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

#ifndef __CUDA_BAYER_H__
#define __CUDA_BAYER_H__


#include "cudaUtility.h"
#include "imageFormat.h"


//////////////////////////////////////////////////////////////////////////////////
/// @name 8-bit Bayer to RGB/RGBA
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Demosaick an 8-bit Bayer image to uchar3 RGB.
 * @params format the Bayer pattern of the input image, should be one of: 	
 *                IMAGE_BAYER_BGGR, IMAGE_BAYER_GBRG, IMAGE_BAYER_GRBG, IMAGE_BAYER_RGGB
 */
cudaError_t cudaBayerToRGB( uint8_t* input, uchar3* output, size_t width, size_t height, imageFormat format );

/**
 * Demosaick an 8-bit Bayer image to uchar4 RGBA.
 * @params format the Bayer pattern of the input image, should be one of: 	
 *                IMAGE_BAYER_BGGR, IMAGE_BAYER_GBRG, IMAGE_BAYER_GRBG, IMAGE_BAYER_RGGB
 */
cudaError_t cudaBayerToRGBA( uint8_t* input, uchar3* output, size_t width, size_t height, imageFormat format );

///@}

#endif

