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

#include "cudaResize.h"



// gpuResize. nearest.
template<typename T>
__global__ void gpuResize_nearest( float2 scale, T* input, int iWidth, T* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const T px = input[ dy * iWidth + dx ];

	output[y*oWidth+x] = px;
}

// gpuResize. linear.
template <typename T>
__global__ void gpuResize_linear( float2 scale, T* input, int iWidth, int iHeight, T* output, int oWidth, int oHeight )
{
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < oWidth && dst_y < oHeight)
	{
		const float src_x = dst_x * scale.x;
		const float src_y = dst_y * scale.y;

		const int x1 = __float2int_rd(src_x);
		const int y1 = __float2int_rd(src_y);
		const int x2 = x1 + 1;
		const int y2 = y1 + 1;
		const int x2_read = ::min(x2, iWidth - 1);
		const int y2_read = ::min(y2, iHeight - 1);

		float4 pix_in[4];
		float weight[4];

		pix_in[0] = make_float4(input[y1 * iWidth + x1]);
		weight[0] = (x2 - src_x) * (y2 - src_y);

		pix_in[1] = make_float4(input[y1 * iWidth + x2_read]);
		weight[1] = (src_x - x1) * (y2 - src_y);

		pix_in[2] = make_float4(input[y2_read * iWidth + x1]);
		weight[2] = (x2 - src_x) * (src_y - y1);

		pix_in[3] = make_float4(input[y2_read * iWidth + x2_read]);
		weight[3] = (src_x - x1) * (src_y - y1);

		float4 out = pix_in[0] * weight[0] + pix_in[1] * weight[1] + pix_in[2] * weight[2] + pix_in[3] * weight[3];

		output[dst_y * oWidth + dst_x] = cast_vec<T>(out);
	}
}

// gpuResize. area.
template <typename T>
__global__ void gpuResize_area( float2 scale, T* input, int iWidth, int iHeight, T* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	{
		float fsx1 = x * scale.x;
		float fsx2 = ::min(fsx1 + scale.x, iWidth - 1.0f);

		int sx1 = __float2int_ru(fsx1);
		int sx2 = __float2int_rd(fsx2);

		float fsy1 = y * scale.y;
		float fsy2 = ::min(fsy1 + scale.y, iHeight - 1.0f);

		int sy1 = __float2int_ru(fsy1);
		int sy2 = __float2int_rd(fsy2);

		float sx = ::min(float(scale.x), iWidth - fsx1);
		float sy = ::min(float(scale.y), iHeight - fsy1);
		float scale = 1.f / (sx * sy);

		float4 out = {};

		for (int dy = sy1; dy < sy2; ++dy)
		{
			// inner rectangle.
			for (int dx = sx1; dx < sx2; ++dx)
				out += make_float4(input[dy * iWidth + dx]) * scale;

			// v-edge line(left).
			if (sx1 > fsx1)
				out += make_float4(input[dy * iWidth + (sx1 -1)]) * ((sx1 - fsx1) * scale);

			// v-edge line(right).
			if (sx2 < fsx2)
				out += make_float4(input[dy * iWidth + sx2]) * ((fsx2 -sx2) * scale);
		}

		// h-edge line(top).
		if (sy1 > fsy1)
			for (int dx = sx1; dx < sx2; ++dx)
				out += make_float4(input[(sy1 - 1) * iWidth + dx]) * ((sy1 -fsy1) * scale);

		// h-edge line(bottom).
		if (sy2 < fsy2)
			for (int dx = sx1; dx < sx2; ++dx)
				out += make_float4(input[sy2 * iWidth + dx]) * ((fsy2 -sy2) * scale);

		// corner(top, left).
		if ((sy1 > fsy1) &&  (sx1 > fsx1))
			out += make_float4(input[(sy1 - 1) * iWidth + (sx1 - 1)]) * ((sy1 -fsy1) * (sx1 -fsx1) * scale);

		// corner(top, right).
		if ((sy1 > fsy1) &&  (sx2 < fsx2))
			out += make_float4(input[(sy1 - 1) * iWidth + sx2]) * ((sy1 -fsy1) * (fsx2 -sx2) * scale);

		// corner(bottom, left).
		if ((sy2 < fsy2) &&  (sx2 < fsx2))
			out += make_float4(input[sy2 * iWidth + sx2]) * ((fsy2 -sy2) * (fsx2 -sx2) * scale);

		// corner(bottom, right).
		if ((sy2 < fsy2) &&  (sx1 > fsx1))
			out += make_float4(input[sy2 * iWidth + (sx1 - 1)]) * ((fsy2 -sy2) * (sx1 -fsx1) * scale);

		output[y * oWidth + x] = cast_vec<T>(out);
	}
}

// launchResize
template<typename T>
static cudaError_t launchResize( T* input, size_t inputWidth, size_t inputHeight,
				             T* output, size_t outputWidth, size_t outputHeight, int mode )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							     float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	if (mode == static_cast<int>(InterpolationFlags::INTER_LINEAR)) {
		if (inputWidth == outputWidth && inputHeight == outputHeight) {
			gpuResize_nearest<T><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);
		} else {
			gpuResize_linear<T><<<gridDim, blockDim>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight);
		}
	} else if (mode == static_cast<int>(InterpolationFlags::INTER_AREA)) {
		if (inputWidth == outputWidth && inputHeight == outputHeight) {
			gpuResize_nearest<T><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);
		} else if (inputWidth < outputWidth || inputHeight < outputHeight) {
			gpuResize_linear<T><<<gridDim, blockDim>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight);
		} else {
			gpuResize_area<T><<<gridDim, blockDim>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight);
		}
	} else {
		gpuResize_nearest<T><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);
	}

	return CUDA(cudaGetLastError());
}

// cudaResize (uint8 grayscale)
cudaError_t cudaResize( uint8_t* input, size_t inputWidth, size_t inputHeight, uint8_t* output, size_t outputWidth, size_t outputHeight, int mode )
{
	return launchResize<uint8_t>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode);
}

// cudaResize (float grayscale)
cudaError_t cudaResize( float* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, int mode )
{
	return launchResize<float>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode);
}

// cudaResize (uchar3)
cudaError_t cudaResize( uchar3* input, size_t inputWidth, size_t inputHeight, uchar3* output, size_t outputWidth, size_t outputHeight, int mode )
{
	return launchResize<uchar3>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode);
}

// cudaResize (uchar4)
cudaError_t cudaResize( uchar4* input, size_t inputWidth, size_t inputHeight, uchar4* output, size_t outputWidth, size_t outputHeight, int mode )
{
	return launchResize<uchar4>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode);
}

// cudaResize (float3)
cudaError_t cudaResize( float3* input, size_t inputWidth, size_t inputHeight, float3* output, size_t outputWidth, size_t outputHeight, int mode )
{
	return launchResize<float3>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode);
}

// cudaResize (float4)
cudaError_t cudaResize( float4* input, size_t inputWidth, size_t inputHeight, float4* output, size_t outputWidth, size_t outputHeight, int mode )
{
	return launchResize<float4>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode);
}

//-----------------------------------------------------------------------------------
cudaError_t cudaResize( void* input,  size_t inputWidth,  size_t inputHeight,
				    void* output, size_t outputWidth, size_t outputHeight, imageFormat format, int mode )
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
		return cudaResize((uchar3*)input, inputWidth, inputHeight, (uchar3*)output, outputWidth, outputHeight, mode);
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return cudaResize((uchar4*)input, inputWidth, inputHeight, (uchar4*)output, outputWidth, outputHeight, mode);
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return cudaResize((float3*)input, inputWidth, inputHeight, (float3*)output, outputWidth, outputHeight, mode);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return cudaResize((float4*)input, inputWidth, inputHeight, (float4*)output, outputWidth, outputHeight, mode);
	else if( format == IMAGE_GRAY8 )
		return cudaResize((uint8_t*)input, inputWidth, inputHeight, (uint8_t*)output, outputWidth, outputHeight, mode);
	else if( format == IMAGE_GRAY32F )
		return cudaResize((float*)input, inputWidth, inputHeight, (float*)output, outputWidth, outputHeight, mode);

	LogError(LOG_CUDA "cudaResize() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");
	LogError(LOG_CUDA "                    * rgb8, bgr8\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgb32f, bgr32f\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}




