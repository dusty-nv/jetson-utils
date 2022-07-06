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
__global__ void gpuResize_linear( float2 scale, T* input, int iWidth, int iHeight, T* output, int oWidth, int oHeight, float max_value )
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
		out = clamp(out, 0.0f, max_value);

		output[dst_y * oWidth + dst_x] = cast_vec<T>(out);
	}
}

// gpuResize. cubic.
template <typename T>
static __device__ T calc_cubic_coef(T d, T a)
{
	T d_abs = abs(d);

	// T w1 = T(1) - (a + T(3)) * d_abs * d_abs + (a + T(2)) * d_abs * d_abs * d_abs;
	// T w2 = T(-4) * a + T(8) * a * d_abs + T(-5) * a * d_abs * d_abs + a * d_abs * d_abs * d_abs;
	T w1 = T(1) + ((a + T(-3)) + (a + T(2)) * d_abs) * d_abs * d_abs;
	T w2 = (T(-4) + (T(8) + (T(-5) + d_abs) * d_abs) * d_abs) * a;

	T w = (d_abs < T(1)) ? w1 : (d_abs < T(2)) ? w2 : T(0);

	return w;
}
template <typename T>
__global__ void gpuResize_cubic( float2 scale, T* input, int iWidth, int iHeight, T* output, int oWidth, int oHeight, float max_value )
{
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < oWidth && dst_y < oHeight)
	{
		const float src_x = dst_x * scale.x;
		const float src_y = dst_y * scale.y;

		const int xc = __float2int_rd(src_x);
		const int yc = __float2int_rd(src_y);
		const int x_btm = xc - 1;
		const int y_btm = yc - 1;
		// const int x_top = xc + 2;
		const int y_top = yc + 2;
		const float xd = src_x - xc;
		const float yd = src_y - yc;

		constexpr float a = -0.75f;	// CPU ver. = -0.75, GPU ver. = -0.5
		const float wx[4] = {
			calc_cubic_coef(-1.0f - xd, a),
			calc_cubic_coef(0.0f - xd, a),
			calc_cubic_coef(1.0f - xd, a),
			calc_cubic_coef(2.0f - xd, a),
		};
		const float wx_sum = wx[0] + wx[1] + wx[2] + wx[3];

		float4 pix_sum = {};
		float w_sum = 0.0f;
		for (int i = y_btm; i <= y_top; i++) {
			const float wy = calc_cubic_coef(i - yd - yc, a);

			const int pos_x[4] = {
				::max(x_btm, 0),
				x_btm + 1,
				::min(x_btm + 2, iWidth - 1),
				::min(x_btm + 3, iWidth - 1),
			};
			const int pos_y = ::clamp(i, 0, iHeight - 1);

			const float4 pix[4] = {
				make_float4(input[pos_y * iWidth + pos_x[0]]),
				make_float4(input[pos_y * iWidth + pos_x[1]]),
				make_float4(input[pos_y * iWidth + pos_x[2]]),
				make_float4(input[pos_y * iWidth + pos_x[3]]),
			};

			pix_sum += (pix[0] * wx[0] + pix[1] * wx[1] + pix[2] * wx[2] + pix[3] * wx[3]) * wy;
			w_sum += wx_sum * wy;
		}

		const float4 out = (w_sum == 0.0f) ? float4{} : clamp(pix_sum / w_sum, 0.0f, max_value);

		output[dst_y * oWidth + dst_x] = cast_vec<T>(out);
	}
}

// gpuResize. area.
template <typename T>
__global__ void gpuResize_area( float2 scale, T* input, int iWidth, int iHeight, T* output, int oWidth, int oHeight, float max_value )
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
				out += make_float4(input[dy * iWidth + dx]);

			// v-edge line(left).
			if (sx1 > fsx1)
				out += make_float4(input[dy * iWidth + (sx1 -1)]) * (sx1 - fsx1);

			// v-edge line(right).
			if (sx2 < fsx2)
				out += make_float4(input[dy * iWidth + sx2]) * (fsx2 - sx2);
		}

		// h-edge line(top).
		if (sy1 > fsy1)
			for (int dx = sx1; dx < sx2; ++dx)
				out += make_float4(input[(sy1 - 1) * iWidth + dx]) * (sy1 - fsy1);

		// h-edge line(bottom).
		if (sy2 < fsy2)
			for (int dx = sx1; dx < sx2; ++dx)
				out += make_float4(input[sy2 * iWidth + dx]) * (fsy2 - sy2);

		// corner(top, left).
		if ((sy1 > fsy1) &&  (sx1 > fsx1))
			out += make_float4(input[(sy1 - 1) * iWidth + (sx1 - 1)]) * (sy1 - fsy1) * (sx1 - fsx1);

		// corner(top, right).
		if ((sy1 > fsy1) &&  (sx2 < fsx2))
			out += make_float4(input[(sy1 - 1) * iWidth + sx2]) * (sy1 - fsy1) * (fsx2 - sx2);

		// corner(bottom, left).
		if ((sy2 < fsy2) &&  (sx2 < fsx2))
			out += make_float4(input[sy2 * iWidth + sx2]) * (fsy2 - sy2) * (fsx2 - sx2);

		// corner(bottom, right).
		if ((sy2 < fsy2) &&  (sx1 > fsx1))
			out += make_float4(input[sy2 * iWidth + (sx1 - 1)]) * (fsy2 - sy2) * (sx1 - fsx1);

		output[y * oWidth + x] = cast_vec<T>(clamp(out * scale, 0.0f, max_value));
	}
}

// gpuResize. lanczos4.
//// /* # SLOW. */
// template <typename T>
// static __device__ T calc_sinc(T x)
// {
// 	return (x == T(0)) ? T(1) : sin(x * M_PI) / (x * M_PI);
// }
// template <typename T>
// static __device__ T calc_lanczos_coef(T d, T n)
// {
// 	T d_abs = abs(d);
// 	T w = (d_abs > n) ? T(0) : calc_sinc(d_abs) * calc_sinc(d_abs / n);

// 	return w;
// }
//// /* # FAST. */
template <typename T>
static __device__ T calc_lanczos_coef(T d, T n)
{
	T d_abs = abs(d);

	T pi_d = M_PI * d_abs;
	T cos_k1 = cos(pi_d * 0.25f);
	T cos_k2 = sqrt(1.0f - cos_k1 * cos_k1);

	T w = (d_abs >= T(4)) ? T(0) :
		(d_abs < T(1e-4f)) ? T(1) :
		(T(16) * cos_k1 * cos_k2 * (cos_k1 - cos_k2) * (cos_k1 + cos_k2) * cos_k2) / (pi_d * pi_d);

	return w;
}
template <typename T>
__global__ void gpuResize_lanczos4( float2 scale, T* input, int iWidth, int iHeight, T* output, int oWidth, int oHeight, float max_value )
{
	constexpr float tap = 4.0f;
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < oWidth && dst_y < oHeight)
	{
		const float src_x = dst_x * scale.x;
		const float src_y = dst_y * scale.y;

		const int xc = __float2int_rd(src_x);
		const int yc = __float2int_rd(src_y);
		const int x_btm = xc - 3;
		const int y_btm = yc - 3;
		// const int x_top = xc + 4;
		const int y_top = yc + 4;
		const float xd = src_x - xc;
		const float yd = src_y - yc;

		const float wx[8] = {
			calc_lanczos_coef(-3.0f - xd, tap),
			calc_lanczos_coef(-2.0f - xd, tap),
			calc_lanczos_coef(-1.0f - xd, tap),
			calc_lanczos_coef(0.0f - xd, tap),
			calc_lanczos_coef(1.0f - xd, tap),
			calc_lanczos_coef(2.0f - xd, tap),
			calc_lanczos_coef(3.0f - xd, tap),
			calc_lanczos_coef(4.0f - xd, tap),
		};
		const float wx_sum = wx[0] + wx[1] + wx[2] + wx[3] + wx[4] + wx[5] + wx[6] + wx[7];

		float4 pix_sum = {};
		float w_sum = 0.0f;
		for (int i = y_btm; i <= y_top; i++) {
			const float wy = calc_lanczos_coef(i - yd - yc, tap);

			const int pos_x[8] = {
				::max(x_btm, 0),
				::max(x_btm + 1, 0),
				::max(x_btm + 2, 0),
				x_btm + 3,
				::min(x_btm + 4, iWidth - 1),
				::min(x_btm + 5, iWidth - 1),
				::min(x_btm + 6, iWidth - 1),
				::min(x_btm + 7, iWidth - 1),
			};
			const int pos_y = ::clamp(i, 0, iHeight - 1);

			const float4 pix[8] = {
				make_float4(input[pos_y * iWidth + pos_x[0]]),
				make_float4(input[pos_y * iWidth + pos_x[1]]),
				make_float4(input[pos_y * iWidth + pos_x[2]]),
				make_float4(input[pos_y * iWidth + pos_x[3]]),
				make_float4(input[pos_y * iWidth + pos_x[4]]),
				make_float4(input[pos_y * iWidth + pos_x[5]]),
				make_float4(input[pos_y * iWidth + pos_x[6]]),
				make_float4(input[pos_y * iWidth + pos_x[7]]),
			};

			const float4 pix_0 = pix[0] * wx[0] + pix[1] * wx[1] + pix[2] * wx[2] + pix[3] * wx[3];
			const float4 pix_1 = pix[4] * wx[4] + pix[5] * wx[5] + pix[6] * wx[6] + pix[7] * wx[7];
			pix_sum += (pix_0 + pix_1) * wy;
			w_sum += wx_sum * wy;
		}

		const float4 out = (w_sum == 0.0f) ? float4{} : clamp(pix_sum / w_sum, 0.0f, max_value);

		output[dst_y * oWidth + dst_x] = cast_vec<T>(out);
	}
}

// gpuResize. spline36.
template <typename T>
static __device__ T calc_spline36_coef(T d)
{
	T d_abs = abs(d);

	T w = (d_abs > T(3)) ? T(0)
		: (d_abs > T(2)) ? (((T(19) * d_abs + T(-159)) * d_abs + T(434)) * d_abs + T(-384)) / T(209)
		: (d_abs > T(1)) ? (((T(-114) * d_abs + T(612)) * d_abs + T(-1038)) * d_abs + T(540)) / T(209)
		: (((T(247) * d_abs + T(-453)) * d_abs + T(-3)) * d_abs + T(209)) / T(209);

	return w;
}
template <typename T>
__global__ void gpuResize_spline36( float2 scale, T* input, int iWidth, int iHeight, T* output, int oWidth, int oHeight, float max_value )
{
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < oWidth && dst_y < oHeight)
	{
		const float src_x = dst_x * scale.x;
		const float src_y = dst_y * scale.y;

		const int xc = __float2int_rd(src_x);
		const int yc = __float2int_rd(src_y);
		const int x_btm = xc - 2;
		const int y_btm = yc - 2;
		// const int x_top = xc + 3;
		const int y_top = yc + 3;
		const float xd = src_x - xc;
		const float yd = src_y - yc;

		const float wx[6] = {
			calc_spline36_coef(-2.0f - xd),
			calc_spline36_coef(-1.0f - xd),
			calc_spline36_coef(0.0f - xd),
			calc_spline36_coef(1.0f - xd),
			calc_spline36_coef(2.0f - xd),
			calc_spline36_coef(3.0f - xd),
		};
		const float wx_sum = wx[0] + wx[1] + wx[2] + wx[3] + wx[4] + wx[5];

		float4 pix_sum = {};
		float w_sum = 0.0f;
		for (int i = y_btm; i <= y_top; i++) {
			const float wy = calc_spline36_coef(i - yd - yc);

			const int pos_x[6] = {
				::max(x_btm, 0),
				::max(x_btm + 1, 0),
				x_btm + 2,
				::min(x_btm + 3, iWidth - 1),
				::min(x_btm + 4, iWidth - 1),
				::min(x_btm + 5, iWidth - 1),
			};
			const int pos_y = ::clamp(i, 0, iHeight - 1);

			const float4 pix[6] = {
				make_float4(input[pos_y * iWidth + pos_x[0]]),
				make_float4(input[pos_y * iWidth + pos_x[1]]),
				make_float4(input[pos_y * iWidth + pos_x[2]]),
				make_float4(input[pos_y * iWidth + pos_x[3]]),
				make_float4(input[pos_y * iWidth + pos_x[4]]),
				make_float4(input[pos_y * iWidth + pos_x[5]]),
			};

			const float4 pix6 = pix[0] * wx[0] + pix[1] * wx[1] + pix[2] * wx[2]
								+ pix[3] * wx[3] + pix[4] * wx[4] + pix[5] * wx[5];
			pix_sum += pix6 * wy;
			w_sum += wx_sum * wy;
		}

		const float4 out = (w_sum == 0.0f) ? float4{} : clamp(pix_sum / w_sum, 0.0f, max_value);

		output[dst_y * oWidth + dst_x] = cast_vec<T>(out);
	}
}

// launchResize
template<typename T>
static cudaError_t launchResize( T* input, size_t inputWidth, size_t inputHeight,
				             T* output, size_t outputWidth, size_t outputHeight, int mode, float max_value )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							     float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(32, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	if (mode == static_cast<int>(InterpolationFlags::INTER_LINEAR)) {
		if (inputWidth == outputWidth && inputHeight == outputHeight) {
			gpuResize_nearest<T><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);
		} else {
			gpuResize_linear<T><<<gridDim, blockDim>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);
		}

	} else if (mode == static_cast<int>(InterpolationFlags::INTER_CUBIC)) {
		if (inputWidth == outputWidth && inputHeight == outputHeight) {
			gpuResize_nearest<T><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);
		} else {
			gpuResize_cubic<T><<<gridDim, blockDim>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);
		}

	} else if (mode == static_cast<int>(InterpolationFlags::INTER_AREA)) {
		if (inputWidth == outputWidth && inputHeight == outputHeight) {
			gpuResize_nearest<T><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);
		} else if (inputWidth < outputWidth || inputHeight < outputHeight) {
			gpuResize_linear<T><<<gridDim, blockDim>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);
		} else {
			gpuResize_area<T><<<gridDim, blockDim>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);
		}

	} else if (mode == static_cast<int>(InterpolationFlags::INTER_LANCZOS4)) {
		if (inputWidth == outputWidth && inputHeight == outputHeight) {
			gpuResize_nearest<T><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);
		} else {
			gpuResize_lanczos4<T><<<gridDim, blockDim>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);
		}

	} else if (mode == static_cast<int>(InterpolationFlags::INTER_SPLINE36)) {
		if (inputWidth == outputWidth && inputHeight == outputHeight) {
			gpuResize_nearest<T><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);
		} else {
			gpuResize_spline36<T><<<gridDim, blockDim>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);
		}

	} else {
		gpuResize_nearest<T><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);
	}

	return CUDA(cudaGetLastError());
}

// cudaResize (uint8 grayscale)
cudaError_t cudaResize( uint8_t* input, size_t inputWidth, size_t inputHeight, uint8_t* output, size_t outputWidth, size_t outputHeight, int mode, float max_value )
{
	return launchResize<uint8_t>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode, max_value);
}

// cudaResize (float grayscale)
cudaError_t cudaResize( float* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, int mode, float max_value )
{
	return launchResize<float>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode, max_value);
}

// cudaResize (uchar3)
cudaError_t cudaResize( uchar3* input, size_t inputWidth, size_t inputHeight, uchar3* output, size_t outputWidth, size_t outputHeight, int mode, float max_value )
{
	return launchResize<uchar3>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode, max_value);
}

// cudaResize (uchar4)
cudaError_t cudaResize( uchar4* input, size_t inputWidth, size_t inputHeight, uchar4* output, size_t outputWidth, size_t outputHeight, int mode, float max_value )
{
	return launchResize<uchar4>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode, max_value);
}

// cudaResize (float3)
cudaError_t cudaResize( float3* input, size_t inputWidth, size_t inputHeight, float3* output, size_t outputWidth, size_t outputHeight, int mode, float max_value )
{
	return launchResize<float3>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode, max_value);
}

// cudaResize (float4)
cudaError_t cudaResize( float4* input, size_t inputWidth, size_t inputHeight, float4* output, size_t outputWidth, size_t outputHeight, int mode, float max_value )
{
	return launchResize<float4>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode, max_value);
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




