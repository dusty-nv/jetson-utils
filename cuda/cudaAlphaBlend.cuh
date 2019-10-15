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

#ifndef __CUDA_ALPHA_BLEND_CUH__
#define __CUDA_ALPHA_BLEND_CUH__


#include "cudaUtility.h"


/**
 * CUDA device function for alpha blending two pixels.
 * The alpha blending of the `src` and `dst` pixels is
 * computed by the equation: `dst * dst.w + src * (1.0 - dst.w)`
 *
 * @note cudaAlphaBlend() is for use inside of other CUDA kernels.
 * @ingroup cuda
 */
__device__ inline float4 cudaAlphaBlend( const float4& src, const float4& dst )
{
	const float alph = dst.w / 255.0f;
	const float inva = 1.0f - alph;

	return make_float4(alph * dst.x + inva * src.x,
				    alph * dst.y + inva * src.y,
				    alph * dst.z + inva * src.z,
				    255.0f);
}


#endif


