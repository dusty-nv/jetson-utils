#include "cudaSplit.h"



// gpuSplit.
// * Split an image on the GPU (supports RGB/BGR, RGBA/BGRA to some single color planes(using GRAY format))
template<typename T, int CH>
__global__ void gpuSplit(T *input, T *output0, T *output1, T *output2, T *output3, size_t width, size_t height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T p0 = input[(y * width + x) * CH + 0];
	const T p1 = input[(y * width + x) * CH + 1];
	const T p2 = input[(y * width + x) * CH + 2];
	const T p3 = (CH == 4) ? input[(y * width + x) * CH + 3] : T(0);

	output0[y * width + x] = p0;
	output1[y * width + x] = p1;
	output2[y * width + x] = p2;
	if (CH == 4) output3[y * width + x] = p3;
}

// gpuSplit.
// * Split an image on the GPU (supports RGBA/BGRA to 3 colors and alpha plane)
template<typename T, typename S, typename R>
__global__ void gpuSplit(T *input, S *output_color, R *output_alpha, size_t width, size_t height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	T pix4 = input[y * width + x];	// 3 colors + alpha.

	S pix3 = cast_vec<S>(pix4);	// 3 colors.
	R pix_a = alpha<T>(pix4);	// alpha.

	output_color[y * width + x] = pix3;
	output_alpha[y * width + x] = pix_a;
}

// launchSplit
// * Split an image on the GPU (supports RGB/BGR, RGBA/BGRA to some single color planes(using GRAY format))
template<typename T, int CH>
static cudaError_t launchSplit(T *input, T **output, size_t width, size_t height)
{
	if( !input || !output[0] || !output[1] || !output[2] || (CH == 4 ? !output[3] : false) )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(32, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuSplit<T, CH><<<gridDim, blockDim>>>(input, output[0], output[1], output[2], (CH == 4) ? output[3] : nullptr, width, height);
	// std::vector<T *> out_ary;
	// for (int i = 0; i < CH; i++) out_ary.push_back(output[i]);
	// printf("=====================\n");
	// gpuSplit<T, CH><<<gridDim, blockDim>>>(input, out_ary, width, height);

	return CUDA(cudaGetLastError());
}

// launchSplit
// * Split an image on the GPU (supports RGBA/BGRA to 3 colors and alpha plane)
template<typename T, typename S, typename R>
static cudaError_t launchSplit(T *input, S *output_color, R *output_alpha, size_t width, size_t height)
{
	if( !input || !output_color || !output_alpha )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(32, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuSplit<T, S, R><<<gridDim, blockDim>>>(input, output_color, output_alpha, width, height);

	return CUDA(cudaGetLastError());
}

//-----------------------------------------------------------------------------------
cudaError_t cudaSplit(void *input, void **output, size_t width, size_t height, imageFormat format)
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
		return launchSplit<uchar, 3>((uchar *)input, (uchar **)output, width, height);
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return launchSplit<uchar, 4>((uchar *)input, (uchar **)output, width, height);
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return launchSplit<float, 3>((float *)input, (float **)output, width, height);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return launchSplit<float, 4>((float *)input, (float **)output, width, height);

	LogError(LOG_CUDA "cudaSplit() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * rgb8, bgr8\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgb32f, bgr32f\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}

cudaError_t cudaSplit(void *input, void *output_color, void *output_alpha, size_t width, size_t height, imageFormat format)
{
	if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return launchSplit<uchar4, uchar3, uchar>((uchar4 *)input, (uchar3 *)output_color, (uchar *)output_alpha, width, height);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return launchSplit<float4, float3, float>((float4 *)input, (float3 *)output_color, (float *)output_alpha, width, height);

	LogError(LOG_CUDA "cudaSplit() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}
