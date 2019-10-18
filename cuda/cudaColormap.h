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

#ifndef __CUDA_COLORMAP_H__
#define __CUDA_COLORMAP_H__


#include "cudaFilterMode.h"


/**
 * Enumeration of built-in colormaps.
 * @see cudaColormapTypeFromStr() and cudaColormapTypeToStr()
 * @ingroup colormap
 */
enum cudaColormapType
{
	// Palettized
	COLORMAP_INFERNO,   /**< Inferno colormap, see http://bids.github.io/colormap/ */
	COLORMAP_MAGMA,     /**< Magma colormap,   see http://bids.github.io/colormap/ */
	COLORMAP_PARULA,    /**< Parula colormap,  see https://www.mathworks.com/help/matlab/ref/parula.html */
	COLORMAP_PLASMA,    /**< Plasma colormap,  see http://bids.github.io/colormap/ */
	COLORMAP_TURBO,     /**< Turbo colormap,   see https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html */
	COLORMAP_VIRIDIS,   /**< Viridis colormap, see http://bids.github.io/colormap/ */

	// Parametric
	COLORMAP_FLOW,		/**< Optical flow x/y velocity (2D) */
	COLORMAP_NONE,		/**< Pass-through (no remapping applied) */
	COLORMAP_LINEAR,	/**< Linearly remap the values to [0,255] */

	/**< Default colormap (Viridis) */
	COLORMAP_DEFAULT = COLORMAP_VIRIDIS
};

/**
 * Parse a cudaColormapType enum from a string.
 * @returns The parsed cudaColormapType, or COLORMAP_DEFAULT on error.
 * @ingroup colormap
 */
cudaColormapType cudaColormapFromStr( const char* colormap );

/**
 * Convert a cudaColormapType enum to a string
 * @ingroup colormap
 */
const char* cudaColormapToStr( cudaColormapType colormap );

/**
 * Apply a colormap from an input image or vector field to RGBA (float4).
 * If the input and output dimensions differ, this function will rescale the image
 * using bilinear or nearest-point interpolation as set by the `filter` mode.
 * @param input_range the minimum and maximum values of the input image.
 * @param colormap the colormap to apply (@see cudaColormapType)
 * @param format layout of multi-channel input data (HWC or CHW). 
 * @ingroup colormap
 */
cudaError_t cudaColormap( float* input, float* output, 
					 size_t width, size_t height,
					 const float2& input_range=make_float2(0,255),
                          cudaColormapType colormap=COLORMAP_DEFAULT,
					 cudaDataFormat format=FORMAT_DEFAULT,
					 cudaStream_t stream=NULL);

/**
 * Apply a colormap from an input image or vector field to RGBA (float4).
 * If the input and output dimensions differ, this function will rescale the image
 * using bilinear or nearest-point interpolation as set by the `filter` mode.
 * @param input_range the minimum and maximum values of the input image.
 * @param colormap the colormap to apply (@see cudaColormapType)
 * @param filter the interpolation mode used for rescaling.
 * @param format layout of multi-channel input data (HWC or CHW).
 * @ingroup colormap
 */
cudaError_t cudaColormap( float* input, size_t input_width, size_t input_height,
					 float* output, size_t output_width, size_t output_height,
					 const float2& input_range=make_float2(0,255),
                          cudaColormapType colormap=COLORMAP_DEFAULT,
					 cudaFilterMode filter=FILTER_LINEAR,
					 cudaDataFormat format=FORMAT_DEFAULT,
					 cudaStream_t stream=NULL );

/**
 * Initialize the colormap palettes by allocating them in CUDA memory.
 * @note cudaColormapInit() is automatically called the first time
 *       that any colormap is used, so it needn't be explicitly
 *       called by the user unless they wish to do it at start-up.
 * @returns cudaSuccess on success or if already initialized.
 * @ingroup colormap
 */
cudaError_t cudaColormapInit();

/**
 * Free the colormap palettes after they are done being used.
 * Only needs to be called if the other colormap functions were.
 * @ingroup colormap
 */
cudaError_t cudaColormapFree();

/**
 * Retrieve a CUDA device pointer to one of the colormaps palettes.
 * The pointer will be to a float4 array that contains 256 elements,
 * with each color's RGBA pixel values in the range from 0.0 to 255.0.
 * @note this function will return `NULL` for the parameterized colormaps
 *       like `COLORMAP_FLOW`, it's only valid for the palettized maps
 *       like `COLORMAP_VIRIDIS`, `COLORMAP_PARULA`, ect.
 * @param colormap the colormap palette to return (@see cudaColormapType)
 * @returns CUDA device pointer to the colormap palette, or `NULL` if
 *          an invalid parameterized colormap was requested.
 */
float4* cudaColormapPalette( cudaColormapType colormap );


#endif

