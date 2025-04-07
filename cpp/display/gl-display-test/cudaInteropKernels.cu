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

#include "cudaInteropKernels.h"


__global__ void gpuGeneratePointGrid( PointVertex* points, int N, float world_size, float time )
{
	const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx_z = blockIdx.y * blockDim.y + threadIdx.y;

	if( idx_x >= N || idx_z >= N )
		return;

	const float half_size = world_size * 0.5f;

	const float p_x = float(idx_x) / float(N);
	const float p_z = float(idx_z) / float(N);

	PointVertex vert;

	vert.pos.x = p_x * world_size - half_size;
	vert.pos.z = p_z * world_size - half_size;
	vert.pos.y = sinf(p_x * p_z * world_size + time * 2.0f);

	vert.color.x = 0.0f;
	vert.color.y = p_x * 255.0f;
	vert.color.z = p_z * 255.0f;
	vert.color.w = 255.0f;

	points[idx_z * N + idx_x] = vert;
}


// cudaGeneratePointGrid
cudaError_t cudaGeneratePointGrid( PointVertex* points, uint32_t N, float world_size, float time )
{
	if( !points )
		return cudaErrorInvalidDevicePointer;

	if( N == 0 )
		return cudaErrorInvalidValue;

	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(N,blockDim.x), iDivUp(N,blockDim.y));

	gpuGeneratePointGrid<<<gridDim, blockDim>>>(points, N, world_size, time);

	return CUDA(cudaGetLastError());
}
