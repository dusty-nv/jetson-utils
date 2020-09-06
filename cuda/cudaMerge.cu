#include "cudaMerge.h"



// gpuMerge.
template<typename T, int CH>
__global__ void gpuMerge(T *input0, T *input1, T *input2, T *input3, T *output, size_t width, size_t height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T p0 = input0[y * width + x];
	const T p1 = input1[y * width + x];
	const T p2 = input2[y * width + x];
	const T p3 = (CH == 4) ? input3[y * width + x] : T(0);

	output[(y * width + x) * CH + 0] = p0;
	output[(y * width + x) * CH + 1] = p1;
	output[(y * width + x) * CH + 2] = p2;
	if (CH == 4) output[(y * width + x) * CH + 3] = p3;
}

// launchMerge
template<typename T, int CH>
static cudaError_t launchMerge(T **input, T *output, size_t width, size_t height)
{
	if( !input[0] || !input[1] || !input[2] || (CH == 4 ? !input[3] : false) || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(32, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuMerge<T, CH><<<gridDim, blockDim>>>(input[0], input[1], input[2], (CH == 4) ? input[3] : nullptr, output, width, height);

	return CUDA(cudaGetLastError());
}

//-----------------------------------------------------------------------------------
cudaError_t cudaMerge(void **input, void *output, size_t width, size_t height, imageFormat format)
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
		return launchMerge<uchar, 3>((uchar **)input, (uchar *)output, width, height);
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return launchMerge<uchar, 4>((uchar **)input, (uchar *)output, width, height);
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return launchMerge<float, 3>((float **)input, (float *)output, width, height);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return launchMerge<float, 4>((float **)input, (float *)output, width, height);

	LogError(LOG_CUDA "cudaMerge() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * rgb8, bgr8\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgb32f, bgr32f\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}
