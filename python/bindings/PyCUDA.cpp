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

#include "PyCUDA.h"

#include "cudaMappedMemory.h"




// PyCUDA_FreeMapped
void PyCUDA_FreeMapped( PyObject* capsule )
{
	printf(LOG_PY_UTILS "freeing CUDA mapped memory\n");

	void* ptr = PyCapsule_GetPointer(capsule, CUDA_MAPPED_MEMORY_CAPSULE);

	if( !ptr )
	{
		printf(LOG_PY_UTILS "PyCUDA_FreeMapped() failed to get pointer from PyCapsule container\n");
		return;
	}

	if( CUDA_FAILED(cudaFreeHost(ptr)) )
	{
		printf(LOG_PY_UTILS "failed to free CUDA mapped memory with cudaFreeHost()\n");
		return;
	}
}


// PyCUDA_RegisterMappedMemory
PyObject* PyCUDA_RegisterMappedMemory( void* cpuPtr, void* gpuPtr, bool freeOnDelete )
{
	if( !cpuPtr || !gpuPtr )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "RegisterMappedMemory() was provided NULL memory pointers");
		return NULL;
	}

	if( cpuPtr != gpuPtr )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "RegisterMappedMemory() pointers don't match");
		
		if( freeOnDelete )
			CUDA(cudaFreeHost(cpuPtr));

		return NULL;
	}

	// create capsule object
	PyObject* capsule = PyCapsule_New(cpuPtr, CUDA_MAPPED_MEMORY_CAPSULE, freeOnDelete ? PyCUDA_FreeMapped : NULL);

	if( !capsule )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "RegisterMappedMemory() failed to create PyCapsule container");
		
		if( freeOnDelete )
			CUDA(cudaFreeHost(cpuPtr));

		return NULL;
	}

	return capsule;
}


// PyCUDA_AllocMapped
PyObject* PyCUDA_AllocMapped( PyObject* self, PyObject* args )
{
	int size = 0;

	if( !PyArg_ParseTuple(args, "i", &size) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() failed to parse size argument");
		return NULL;
	}
		
	if( size <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() requested size is negative or zero");
		return NULL;
	}

	// allocate memory
	void* cpuPtr = NULL;
	void* gpuPtr = NULL;

	if( !cudaAllocMapped(&cpuPtr, &gpuPtr, size) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() failed");
		return NULL;
	}

	return PyCUDA_RegisterMappedMemory(cpuPtr, gpuPtr);
}


//-------------------------------------------------------------------------------

static PyMethodDef pyCUDA_Functions[] = 
{
	{ "cudaAllocMapped", (PyCFunction)PyCUDA_AllocMapped, METH_VARARGS, "Allocate CUDA ZeroCopy mapped memory" },
	{NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyCUDA_RegisterFunctions()
{
	return pyCUDA_Functions;
}

// Register types
bool PyCUDA_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;
	
	return true;
}

