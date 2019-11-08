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

#include "cudaCrop.h"



// gpuCrop
template <typename T>
__global__ void gpuCrop( T* input, T* output, int offsetX, int offsetY, 
					int inWidth, int outWidth, int outHeight )
{
	const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

	if( out_x >= outWidth || out_y >= outHeight )
		return;

	const int in_x = out_x + offsetX;
	const int in_y = out_y + offsetY;

	output[out_y * outWidth + out_x] = input[in_y * inWidth + in_x];
}


// cudaCrop
cudaError_t cudaCrop( float* input, float* output, const int4& roi, 
				  size_t inputWidth, size_t inputHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || inputHeight == 0 )
		return cudaErrorInvalidValue;

	// get the ROI/output dimensions
	const int outputWidth = roi.z - roi.x;
	const int outputHeight = roi.w - roi.y;

	// validate the requested ROI
	if( outputWidth <= 0 || outputHeight <= 0 )
		return cudaErrorInvalidValue;

	if( outputWidth > inputWidth || outputHeight > inputHeight )
		return cudaErrorInvalidValue;

	if( roi.x < 0 || roi.y < 0 || roi.z < 0 || roi.w < 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuCrop<float><<<gridDim, blockDim>>>(input, output, roi.x, roi.y, inputWidth, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}


// cudaCropRGBA
cudaError_t cudaCropRGBA( float4* input, float4* output, const int4& roi, 
					 size_t inputWidth, size_t inputHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || inputHeight == 0 )
		return cudaErrorInvalidValue;

	// get the ROI/output dimensions
	const int outputWidth = roi.z - roi.x;
	const int outputHeight = roi.w - roi.y;

	// validate the requested ROI
	if( outputWidth <= 0 || outputHeight <= 0 )
		return cudaErrorInvalidValue;

	if( outputWidth > inputWidth || outputHeight > inputHeight )
		return cudaErrorInvalidValue;

	if( roi.x < 0 || roi.y < 0 || roi.z < 0 || roi.w < 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuCrop<float4><<<gridDim, blockDim>>>(input, output, roi.x, roi.y, inputWidth, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}





