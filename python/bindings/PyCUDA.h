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
} PyCudaMemory;

// PyCudaImage object
typedef struct {
	PyCudaMemory base;
	imageFormat format;
	uint32_t    width;
	uint32_t    height;
	Py_ssize_t  shape[3];
	Py_ssize_t  strides[3];
} PyCudaImage;

// Create memory objects
PyObject* PyCUDA_RegisterMemory( void* ptr, size_t size, bool mapped=false, bool freeOnDelete=true );
//PyObject* PyCUDA_RegisterMappedMemory( void* ptr, size_t size, bool freeOnDelete=true );

// Create image objects
PyObject* PyCUDA_RegisterImage( void* ptr, uint32_t width, uint32_t height, imageFormat format, bool mapped=false, bool freeOnDelete=true );
//PyObject* PyCUDA_RegisterMappedImage( void* ptr, uint32_t width, uint32_t height, imageFormat format, bool freeOnDelete=true );

// type checks
bool PyCUDA_IsMemory( PyObject* object );
bool PyCUDA_IsImage( PyObject* object );

// cast operators
PyCudaMemory* PyCUDA_GetMemory( PyObject* object );
PyCudaImage* PyCUDA_GetImage( PyObject* object );

// retrieve from capsule
void* PyCUDA_GetImage( PyObject* object, int* width, int* height, imageFormat* format );

// Register functions
PyMethodDef* PyCUDA_RegisterFunctions();

// Register types
bool PyCUDA_RegisterTypes( PyObject* module );


#endif

