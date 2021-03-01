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

#include "cudaBayer.h"
#include "logging.h"

#include <nppi.h>


// cudaBayerToRGB
cudaError_t cudaBayerToRGB( uint8_t* input, uchar3* output, size_t width, size_t height, imageFormat format )
{
	NppiSize size;
	size.width = width;
	size.height = height;
	
	NppiRect roi;
	roi.x = 0;
	roi.y = 0;
	roi.width = width;
	roi.height = height;
	
	NppiBayerGridPosition grid;
	
	if( format == IMAGE_BAYER_BGGR )
		grid = NPPI_BAYER_BGGR;
	else if( format == IMAGE_BAYER_GBRG )
		grid = NPPI_BAYER_GBRG;
	else if( format == IMAGE_BAYER_GRBG )
		grid = NPPI_BAYER_GRBG;
	else if( format == IMAGE_BAYER_RGGB )
		grid = NPPI_BAYER_RGGB;
	else
		return cudaErrorInvalidValue;
	
	const NppStatus result = nppiCFAToRGB_8u_C1C3R(input, width * sizeof(uint8_t), size, roi, 
												   (uint8_t*)output, width * sizeof(uchar3),
												   grid, NPPI_INTER_UNDEFINED);
	
	if( result != 0 )
	{
		LogError(LOG_CUDA "cudaBayerToRGB() NPP error %i\n", result);
		return cudaErrorUnknown;
	}
	return cudaSuccess;
}


cudaError_t cudaBayerToRGBA( uint8_t* input, uchar3* output, size_t width, size_t height, imageFormat format )
{
	return cudaErrorInvalidValue;
	
}



