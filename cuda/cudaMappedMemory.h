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

#ifndef __CUDA_MAPPED_MEMORY_H_
#define __CUDA_MAPPED_MEMORY_H_


#include "cudaUtility.h"
#include "imageFormat.h"
#include "logging.h"


/**
 * Allocate mapped memory that shares the same physical memory between CPU and GPU,
 * as is the case on Jetson devices with unified memory.  This eliminates the need
 * for memory copies (Zero Copy), although synchronization is still required so that
 * both processors are not accessing the same memory simultaneously.
 *
 * @param[out] ptr Returned pointer to the shared memory, can be accessed from both the CPU
 *                 and in CUDA kernels This memory should be released with cudaFreeHost()
 * @param[in] size Size (in bytes) of the shared memory to allocate.
 * @param[in] clear If `true` (the default), the memory contents will be filled with zeros.
 *
 * @note This function is the same as cudaAllocMapped(), but returns cudaError_t instead of bool.
 * @returns cudaSuccess on success, cudaError_t on failure.
 * @ingroup cudaMemory
 */
inline cudaError_t cudaMallocMapped( void** ptr, size_t size, bool clear=true )
{
	void* cpu = NULL;
	void* gpu = NULL;

	if( !ptr || size == 0 )
		return cudaErrorInvalidValue;

	//CUDA_ASSERT(cudaSetDeviceFlags(cudaDeviceMapHost));

    CUDA_ASSERT(cudaHostAlloc(&cpu, size, cudaHostAllocMapped));
    CUDA_ASSERT(cudaHostGetDevicePointer(&gpu, cpu, 0));

    if( cpu != gpu )
    {
        LogError(LOG_CUDA "cudaMallocMapped() - addresses of CPU and GPU pointers don't match (CPU=%p GPU=%p)\n", cpu, gpu);
        return cudaErrorInvalidDevicePointer;
    }
    
    if( clear )
	    memset(cpu, 0, size);

    *ptr = cpu;
	return cudaSuccess;
}


/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * @note this overload of cudaAllocMapped returns one pointer, assumes that the 
 *       CPU and GPU addresses will match (as is the case with any recent CUDA version).
 *
 * @param[out] ptr Returned pointer to the shared CPU/GPU memory.
 * @param[in] size Size (in bytes) of the shared memory to allocate.
 * @param[in] clear If `true` (default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
inline bool cudaAllocMapped( void** ptr, size_t size, bool clear=true )
{
    return CUDA_SUCCESS(cudaMallocMapped(ptr, size, clear));
}

/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * This overload is for allocating images from an imageFormat type
 * and the image dimensions.  The overall size of the allocation 
 * will be calculated with the imageFormatSize() function.
 *
 * @param[out] ptr Returned pointer to the shared CPU/GPU memory.
 * @param[in] width Width (in pixels) to allocate.
 * @param[in] height Height (in pixels) to allocate. 
 * @param[in] format Format of the image.
 * @param[in] clear If `true` (default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
inline bool cudaAllocMapped( void** ptr, size_t width, size_t height, imageFormat format, bool clear=true )
{
	return cudaAllocMapped(ptr, imageFormatSize(format, width, height), clear);
}


/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * This overload is for allocating images from an imageFormat type
 * and the image dimensions.  The overall size of the allocation 
 * will be calculated with the imageFormatSize() function.
 *
 * @param[out] ptr Returned pointer to the shared CPU/GPU memory.
 * @param[in] dims `int2` vector where `width=dims.x` and `height=dims.y`
 * @param[in] format Format of the image.
 * @param[in] clear If `true` (default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
inline bool cudaAllocMapped( void** ptr, const int2& dims, imageFormat format, bool clear=true )
{
	return cudaAllocMapped(ptr, imageFormatSize(format, dims.x, dims.y), clear);
}


/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * This is a templated version for allocating images from vector types
 * like uchar3, uchar4, float3, float4, ect.  The overall size of the
 * allocation will be calculated as `width * height * sizeof(T)`.
 *
 * @param[out] ptr Returned pointer to the shared CPU/GPU memory.
 * @param[in] width Width (in pixels) to allocate.
 * @param[in] height Height (in pixels) to allocate. 
 * @param[in] clear If `true` (default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
template<typename T> inline bool cudaAllocMapped( T** ptr, size_t width, size_t height, bool clear=true )
{
	return cudaAllocMapped((void**)ptr, width * height * sizeof(T), clear);
}


/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * This is a templated version for allocating images from vector types
 * like uchar3, uchar4, float3, float4, ect.  The overall size of the
 * allocation will be calculated as `dims.x * dims.y * sizeof(T)`.
 *
 * @param[out] ptr Returned pointer to the shared CPU/GPU memory.
 * @param[in] dims `int2` vector where `width=dims.x` and `height=dims.y`
 * @param[in] clear If `true` (default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
template<typename T> inline bool cudaAllocMapped( T** ptr, const int2& dims, bool clear=true )
{
	return cudaAllocMapped((void**)ptr, dims.x * dims.y * sizeof(T), clear);
}


/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * This is a templated version for allocating images from vector types
 * like uchar3, uchar4, float3, float4, ect.  The overall size of the
 * allocation is specified by the size parameter.
 *
 * @param[out] ptr Returned pointer to the shared CPU/GPU memory.
 * @param[in] size size of the allocation, in bytes.
 * @param[in] clear If `true` (default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
template<typename T> inline bool cudaAllocMapped( T** ptr, size_t size, bool clear=true )
{
	return cudaAllocMapped((void**)ptr, size, clear);
}


/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * @note although two pointers are returned, one for CPU and GPU, they both resolve to the same physical memory.
 *
 * @param[out] cpuPtr Returned CPU pointer to the shared memory.
 * @param[out] gpuPtr Returned GPU pointer to the shared memory.
 * @param[in] size Size (in bytes) of the shared memory to allocate.
 * @param[in] clear If `true` (the default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
inline bool cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size, bool clear=true )
{
	if( !cpuPtr || !gpuPtr || size == 0 )
		return false;

	//CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

	if( CUDA_FAILED(cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped)) )
		return false;

	if( CUDA_FAILED(cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0)) )
		return false;

    if( clear )
	    memset(*cpuPtr, 0, size);
	    
	LogDebug(LOG_CUDA "cudaAllocMapped %zu bytes, CPU %p GPU %p\n", size, *cpuPtr, *gpuPtr);
	return true;
}

#endif
