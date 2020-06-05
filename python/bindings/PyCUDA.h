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
 
#ifndef __PYTHON_BINDINGS_CUDA__
#define __PYTHON_BINDINGS_CUDA__

#include "PyUtils.h"


// Name of memory capsules
#define CUDA_MALLOC_MEMORY_CAPSULE	PY_UTILS_MODULE_NAME ".cudaMalloc"
#define CUDA_MAPPED_MEMORY_CAPSULE PY_UTILS_MODULE_NAME ".cudaAllocMapped"

// Create memory capsule
PyObject* PyCUDA_RegisterMemory( void* gpuPtr, bool freeOnDelete=true );

// Create mapped memory capsule
PyObject* PyCUDA_RegisterMappedMemory( void* gpuPtr, bool freeOnDelete=true );
PyObject* PyCUDA_RegisterMappedMemory( void* cpuPtr, void* gpuPtr, bool freeOnDelete=true );

// Register functions
PyMethodDef* PyCUDA_RegisterFunctions();

// Register types
bool PyCUDA_RegisterTypes( PyObject* module );

// Retrieve pointer from capsule object
inline void* PyCUDA_GetPointer( PyObject* capsule ) 
{
	if( PyCapsule_IsValid(capsule, CUDA_MAPPED_MEMORY_CAPSULE) != 0 )
		return PyCapsule_GetPointer(capsule, CUDA_MAPPED_MEMORY_CAPSULE); 
	else if( PyCapsule_IsValid(capsule, CUDA_MALLOC_MEMORY_CAPSULE) )
		return PyCapsule_GetPointer(capsule, CUDA_MALLOC_MEMORY_CAPSULE); 
}

#endif
