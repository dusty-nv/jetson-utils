#include "cudaSplit.h"



// gpuSplit.
// template<typename T, int CH>
// __global__ void gpuSplit(T *input, std::vector<T *> &output, size_t width, size_t height)
// {
// 	const int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	const int y = blockIdx.y * blockDim.y + threadIdx.y;

// 	if( x >= width || y >= height )
// 		return;

// 	// printf("%s:%d: %p\n", __FILE__, __LINE__, input);
// 	const T p0 = input[(y * width + x) * CH + 0];
// 	const T p1 = input[(y * width + x) * CH + 1];
// 	const T p2 = input[(y * width + x) * CH + 2];
// 	const T p3 = (CH == 4) ? input[(y * width + x) * CH + 3] : T(0);
// 	// if (p0 != p1 || p1 != p2 || p2 != p0) {
// 	// 	printf("%s:%d: %d,%d,%d,%d\n", __FILE__, __LINE__, int(p0), int(p1), int(p2), int(p3));
// 	// }

// 	// printf("%s:%d: %p,%p,%p,%p\n", __FILE__, __LINE__, output[0], output[1], output[2], (CH == 4) ? output[3] : nullptr);
// 	// printf("%s:%d: %p\n", __FILE__, __LINE__, output.at(0));
// 	// T *ptr_o0 = output[0];
// 	// T *ptr_o1 = output[1];
// 	// T *ptr_o2 = output[2];
// 	// T *ptr_o3 = (CH == 4) ? output[3] : nullptr;

// 	// printf("%s:%d: %p,%p,%p,%p\n", __FILE__, __LINE__, ptr_o0, ptr_o1, ptr_o2, ptr_o3);
// 	// ptr_o0[y * width + x] = p0;
// 	// ptr_o1[y * width + x] = p1;
// 	// ptr_o2[y * width + x] = p2;
// 	// if (CH == 4 ) ptr_o3[y * width + x] = p3;
// }
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

// launchSplit
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
