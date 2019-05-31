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

#include "PyNumPy.h"
#include "PyCUDA.h"

#include "cudaMappedMemory.h"


#ifdef HAS_NUMPY

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "loadImage.h"



// cudaToNumpy()
PyObject* PyNumPy_FromCUDA( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	PyObject* capsule = NULL;

	int width  = 0;
	int height = 1;
	int depth  = 1;

	static char* kwlist[] = {"array", "width", "height", "depth", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "Oi|ii", kwlist, &capsule, &width, &height, &depth))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaToNumpy() failed to parse args tuple");
		return NULL;
	}

	// verify dimensions
	if( width <= 0 || height <= 0 || depth <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaToNumpy() array dimensions are invalid");
		return NULL;
	}

	// get pointer to image data
	void* src = PyCapsule_GetPointer(capsule, CUDA_MAPPED_MEMORY_CAPSULE);	// TODO  support GPU-only memory

	if( !src )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaToNumpy() failed to get input array pointer from PyCapsule container");
		return NULL;
	}

	// setup dims
	npy_intp dims[] = { height, width, depth };

	// create numpy array
	PyObject* array = PyArray_SimpleNewFromData(3, dims, NPY_FLOAT32, src);

	if( !array )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaToNumpy() failed to create numpy array");
		return NULL;
	}

	// since the numpy array is using the capsule's memory,
	// have it reference the capsule to manage it's deletion
	Py_INCREF(capsule);
	PyArray_SetBaseObject((PyArrayObject*)array, capsule);
	
	// return the numpy array
	return array;
}



// cudaFromNumpy()
PyObject* PyNumPy_ToCUDA( PyObject* self, PyObject* args )
{
	PyObject* object = NULL;

	if( !PyArg_ParseTuple(args, "O", &object) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFromNumpy() failed to parse array argument");
		return NULL;
	}
		
	// cast to numpy array
	PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(object, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY|NPY_ARRAY_FORCECAST);

	if( !array )
		return NULL;

	// calculate the size of the array
	const int ndim = PyArray_NDIM(array);
	npy_intp* dims = PyArray_DIMS(array);
	size_t    size = 0;

	for( int n=0; n < ndim; n++ )
	{
		printf(LOG_PY_UTILS "cudaFromNumpy()  ndarray dim %i = %li\n", n, dims[n]);

		if( n == 0 )
			size = dims[0];
		else
			size *= dims[n];
	}

	size *= sizeof(float);

	if( size == 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFromNumpy() numpy ndarray has data size of 0 bytes");
		Py_DECREF(array);
		return NULL;
	}


	// retrieve the data pointer to the array
	float* arrayPtr = (float*)PyArray_DATA(array);

	if( !arrayPtr )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFromNumpy() failed to retrieve data pointer from numpy ndarray");
		Py_DECREF(array);
		return NULL;
	}

	// allocate CUDA memory for the array
	void* cpuPtr = NULL;
	void* gpuPtr = NULL;

	if( !cudaAllocMapped(&cpuPtr, &gpuPtr, size) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() failed");
		Py_DECREF(array);
		return NULL;
	}	

	// register CUDA memory capsule
	PyObject* capsule = PyCUDA_RegisterMappedMemory(cpuPtr, gpuPtr);

	if( !capsule )
	{
		Py_DECREF(array);
		return NULL;
	}

	// copy array into CUDA memory
	memcpy(cpuPtr, arrayPtr, size);

	// return capsule container
	Py_DECREF(array);
	return capsule;
}



//-------------------------------------------------------------------------------

static PyMethodDef pyImageIO_Functions[] = 
{
	{ "cudaFromNumpy", (PyCFunction)PyNumPy_ToCUDA, METH_VARARGS, "Copy a numpy ndarray to CUDA memory" },
	{ "cudaToNumpy", (PyCFunction)PyNumPy_FromCUDA, METH_VARARGS|METH_KEYWORDS, "Create a numpy ndarray wrapping the CUDA memory, without copying it" },	
	{NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyNumPy_RegisterFunctions()
{
	return pyImageIO_Functions;
}

// Initialize NumPy
PyMODINIT_FUNC PyNumPy_ImportNumPy()
{
	import_array();
	//import_ufunc();	// only needed if using ufunctions
}

// Register types
bool PyNumPy_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;
	
	PyNumPy_ImportNumPy();
	return true;
}

#else

// stub functions
PyMethodDef* PyNumPy_RegisterFunctions()
{
	printf(LOG_PY_UTILS "compiled without NumPy array conversion support (warning)\n");
	printf(LOG_PY_UTILS "if you wish to have support for converting NumPy arrays,\n");
	printf(LOG_PY_UTILS "first run 'sudo apt-get install python-numpy python3-numpy'\n");

	return NULL;
}

// Register types
bool PyNumPy_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;
	
	return true;
}

#endif

