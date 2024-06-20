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
#include "imageFormat.h"

// PyCudaMemory object
typedef struct {
	PyObject_HEAD
	void* ptr;
	size_t size;
	bool mapped;
	bool freeOnDelete;
	cudaStream_t stream;
	cudaEvent_t event;
} PyCudaMemory;

// PyCudaImage object
typedef struct {
	PyCudaMemory base;
	imageFormat format;
	uint64_t    timestamp;
	uint32_t    width;
	uint32_t    height;
	Py_ssize_t  shape[3];
	Py_ssize_t  strides[3];
	PyObject*   cudaArrayInterfaceDict;  // https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
} PyCudaImage;

// Create memory objects
PyObject* PyCUDA_RegisterMemory( void* ptr, size_t size, bool mapped=false, bool freeOnDelete=true );
//PyObject* PyCUDA_RegisterMappedMemory( void* ptr, size_t size, bool freeOnDelete=true );

// Create image objects
PyObject* PyCUDA_RegisterImage( void* ptr, uint32_t width, uint32_t height, imageFormat format, uint64_t timestamp=0, bool mapped=false, bool freeOnDelete=true );
//PyObject* PyCUDA_RegisterMappedImage( void* ptr, uint32_t width, uint32_t height, imageFormat format, bool freeOnDelete=true );

// type checks
bool PyCUDA_IsMemory( PyObject* object );
bool PyCUDA_IsImage( PyObject* object );

// cast operators
PyCudaMemory* PyCUDA_GetMemory( PyObject* object );
PyCudaImage* PyCUDA_GetImage( PyObject* object );

// retrieve from capsule
void* PyCUDA_GetImage( PyObject* object, int* width, int* height, imageFormat* format, uint64_t* timestamp=NULL );

// Register functions
PyMethodDef* PyCUDA_RegisterFunctions();

// Register types
bool PyCUDA_RegisterTypes( PyObject* module );


// Exception handling
#define PYCUDA_ASSERT(x)        PYCUDA_CHECK(x, NULL)
#define PYCUDA_ASSERT_NOGIL(x)  PYCUDA_CHECK_NOGIL(x, NULL)

#define PYCUDA_CHECK(x, return_on_error) { \
    const cudaError_t _retval = cudaCheckError((x), #x, __FILE__, __LINE__); \
    if( _retval != cudaSuccess ) { \
        PyErr_Format(PyExc_Exception, "CUDA error (%u) - %s\n  File \"%s\", line %i\n    %s", _retval, cudaGetErrorString(_retval), __FILE__, __LINE__, #x); \
        return return_on_error; \
    } \
}

#define PYCUDA_CHECK_NOGIL(x, return_on_error) { \
    cudaError_t _retval = cudaSuccess; \
    Py_BEGIN_ALLOW_THREADS \
    _retval = cudaCheckError((x), #x, __FILE__, __LINE__); \
    Py_END_ALLOW_THREADS \
    if( _retval != cudaSuccess ) { \
        PyErr_Format(PyExc_Exception, "CUDA error (%u) - %s\n  File \"%s\", line %i\n    %s", _retval, cudaGetErrorString(_retval), __FILE__, __LINE__, #x); \
        return return_on_error; \
    } \
}

#endif

