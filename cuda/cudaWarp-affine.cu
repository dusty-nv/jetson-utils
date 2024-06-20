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

#include "cudaWarp.h"
#include "mat33.h"


// gpuPerspectiveWarp
template<typename T>
__global__ void gpuPerspectiveWarp( T* input, T* output, int width, int height,
                                    float3 m0, float3 m1, float3 m2 )
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
				   
	if( x >= width || y >= height )
		return;
	
	const float3 vec = make_float3(x, y, 1.0f);
				 
	const float3 vec_out = make_float3( m0.x * vec.x + m0.y * vec.y + m0.z * vec.z,
								 m1.x * vec.x + m1.y * vec.y + m1.z * vec.z,
								 m2.x * vec.x + m2.y * vec.y + m2.z * vec.z );
	
	const int u = vec_out.x / vec_out.z;
	const int v = vec_out.y / vec_out.z;
	
	T px;

	px.x = 0; px.y = 255;
	px.z = 0; px.w = 255;

	if( u < width && v < height && u >= 0 && v >= 0 )
		px = input[v * width + u];
		
     //if( x != u && y != v )
	//	printf("(%i, %i) -> (%i, %i)\n", u, v, x, y);

	output[y * width + x] = px;
} 


// setup the transformation for the CUDA kernel
inline static void invertTransform( float3 cuda_mat[3], const float transform[3][3], bool transform_inverted )
{
	// invert the matrix if it isn't already
	if( !transform_inverted )
	{
		float inv[3][3];

		mat33_inverse(inv, transform);

		for( uint32_t i=0; i < 3; i++ )
		{
			cuda_mat[i].x = inv[i][0];
			cuda_mat[i].y = inv[i][1];
			cuda_mat[i].z = inv[i][2];
		}
	}
	else
	{
		for( uint32_t i=0; i < 3; i++ )
		{
			cuda_mat[i].x = transform[i][0];
			cuda_mat[i].y = transform[i][1];
			cuda_mat[i].z = transform[i][2];
		}
	}
}


// cudaWarpPerspective
cudaError_t cudaWarpPerspective( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
                                 const float transform[3][3], bool transform_inverted, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// setup the transform
	float3 cuda_mat[3];
	invertTransform(cuda_mat, transform, transform_inverted);

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuPerspectiveWarp<<<gridDim, blockDim, 0, stream>>>(input, output, width, height, 
                                                         cuda_mat[0], cuda_mat[1], cuda_mat[2]);

	return CUDA(cudaGetLastError());
}


// cudaWarpPerspective
cudaError_t cudaWarpPerspective( float4* input, float4* output, uint32_t width, uint32_t height,
                                 const float transform[3][3], bool transform_inverted, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// setup the transform
	float3 cuda_mat[3];
	invertTransform(cuda_mat, transform, transform_inverted);

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuPerspectiveWarp<<<gridDim, blockDim, 0, stream>>>(input, output, width, height, 
                                                         cuda_mat[0], cuda_mat[1], cuda_mat[2]);

	return CUDA(cudaGetLastError());
}


// cudaWarpAffine
cudaError_t cudaWarpAffine( float4* input, float4* output, uint32_t width, uint32_t height,
                            const float transform[2][3], bool transform_inverted, cudaStream_t stream )
{
	float psp_transform[3][3];

	// convert the affine transform to 3x3
	for( uint32_t i=0; i < 2; i++ )
		for( uint32_t j=0; j < 3; j++ )
			psp_transform[i][j] = transform[i][j];

	psp_transform[2][0] = 0;
	psp_transform[2][1] = 0;
	psp_transform[2][2] = 1;

	return CUDA(cudaWarpPerspective(input, output, width, height, psp_transform, transform_inverted, stream));
}


// cudaWarpAffine
cudaError_t cudaWarpAffine( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
                            const float transform[2][3], bool transform_inverted, cudaStream_t stream )
{
	float psp_transform[3][3];

	// convert the affine transform to 3x3
	for( uint32_t i=0; i < 2; i++ )
		for( uint32_t j=0; j < 3; j++ )
			psp_transform[i][j] = transform[i][j];

	psp_transform[2][0] = 0;
	psp_transform[2][1] = 0;
	psp_transform[2][2] = 1;

	return CUDA(cudaWarpPerspective(input, output, width, height, psp_transform, transform_inverted, stream));
}


//----------------------------------------------------------------------------------------
// gpuPerspectiveWarp2 (supports different input/output dims)
//----------------------------------------------------------------------------------------
template<typename T>
__global__ void gpuPerspectiveWarp2( T* input, int inputWidth, int inputHeight,
                                     T* output, int outputWidth, int outputHeight,
                                     float3 m0, float3 m1, float3 m2 )
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
				   
	if( x >= outputWidth || y >= outputHeight )
		return;
	
	const float3 vec = make_float3(x, y, 1.0f);
				 
	const float3 vec_out = make_float3( m0.x * vec.x + m0.y * vec.y + m0.z * vec.z,
								 m1.x * vec.x + m1.y * vec.y + m1.z * vec.z,
								 m2.x * vec.x + m2.y * vec.y + m2.z * vec.z );
	
	const int u = vec_out.x / vec_out.z;
	const int v = vec_out.y / vec_out.z;

	if( u < inputWidth && v < inputHeight && u >= 0 && v >= 0 )
		output[y * outputWidth + x] = input[v * inputWidth + u];
} 

cudaError_t cudaWarpPerspective( void* input, uint32_t inputWidth, uint32_t inputHeight, imageFormat inputFormat,
                                 void* output, uint32_t outputWidth, uint32_t outputHeight, imageFormat outputFormat,
                                 const float transform[3][3], bool transform_inverted, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || inputHeight == 0 || outputWidth == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	if( inputFormat != outputFormat )
	{
		LogError(LOG_CUDA "cudaWarpPerspective() -- input and output images must be of the same datatype/format\n");
		return cudaErrorInvalidValue;
	}
	
	// setup the transform
	float3 cuda_mat[3];
	invertTransform(cuda_mat, transform, transform_inverted);

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	#define LAUNCH_PERSPECTIVE_WARP2(type) \
		gpuPerspectiveWarp2<type><<<gridDim, blockDim, 0, stream>>>((type*)input, inputWidth, inputHeight, (type*)output, outputWidth, outputHeight, cuda_mat[0], cuda_mat[1], cuda_mat[2])
	
	if( outputFormat == IMAGE_RGB8 )
		LAUNCH_PERSPECTIVE_WARP2(uchar3);
	else if( outputFormat == IMAGE_RGBA8 )
		LAUNCH_PERSPECTIVE_WARP2(uchar4);
	else if( outputFormat == IMAGE_RGB32F )
		LAUNCH_PERSPECTIVE_WARP2(float3); 
	else if( outputFormat == IMAGE_RGBA32F )
		LAUNCH_PERSPECTIVE_WARP2(float4);
	else
	{
		imageFormatErrorMsg(LOG_CUDA, "cudaWarpPerspective()", outputFormat);
		return cudaErrorInvalidValue;
	}
		
	return cudaGetLastError();
}
						   
