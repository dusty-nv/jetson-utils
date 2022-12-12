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

#include "PyNumpy.h"
#include "PyCUDA.h"

#include "cudaMappedMemory.h"
#include "logging.h"

#ifdef HAS_NUMPY

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "loadImage.h"



// imageFormat to numpy dtype
static int PyNumpy_ConvertFormat( imageFormat format )
{
	const imageBaseType baseType = imageFormatBaseType(format);

	if( baseType == IMAGE_FLOAT )
		return NPY_FLOAT32;
	else if( baseType == IMAGE_UINT8 )
		return NPY_UINT8;

	return NPY_VOID;
}


// cudaToNumpy()
PyObject* PyNumpy_FromCUDA( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	PyObject* capsule = NULL;

	int width  = 0;
	int height = 1;
	int depth  = 1;

	static char* kwlist[] = {"array", "width", "height", "depth", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist, &capsule, &width, &height, &depth))
		return NULL;

	// verify dimensions
	/*if( width <= 0 || height <= 0 || depth <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaToNumpy() array dimensions are invalid");
		return NULL;
	}*/

	// get pointer to image data
	PyCudaImage* img = PyCUDA_GetImage(capsule);
	
	void* src = NULL;
	int type = NPY_FLOAT32;	// float is assumed for PyCudaMemory case, but inferred for PyCudaImage case
	bool mapped = false;
	
	if( !img )
	{
		PyCudaMemory* mem = PyCUDA_GetMemory(capsule);
		
		if( !mem )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaToNumpy() failed to get input CUDA pointer from first arg (should be cudaImage or cudaMemory)");
			return NULL;
		}
		
		src = mem->ptr;
		mapped = mem->mapped;
	}
	else
	{
		if ( imageFormatIsYUV(img->format) )
		{
			src    = img->base.ptr;
			mapped = img->base.mapped;
			width  = 1;
			height = 1;
			depth  = img->base.size;
			type   = PyNumpy_ConvertFormat(img->format);
		}
		else
		{
			src    = img->base.ptr;
			mapped = img->base.mapped;
			width  = img->width;
			height = img->height;
			depth  = imageFormatChannels(img->format);
			type   = PyNumpy_ConvertFormat(img->format);
		}
	}
	
	if( !mapped )   // TODO  support GPU-only memory
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaToNumpy() needs to use CUDA mapped memory as input (allocate with mapped=1)");
		return NULL;
	}
	
	// setup dims
	npy_intp dims[] = { height, width, depth };

	// create numpy array
	PyObject* array = PyArray_SimpleNewFromData(3, dims, type, src);

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
PyObject* PyNumpy_ToCUDA( PyObject* self, PyObject* args, PyObject* kwds )
{
	PyObject* object = NULL;

	int pyBGR=0;
	static char* kwlist[] = {"array", "isBGR", "timestamp", NULL};
	long long timestamp = 0;

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|iL", kwlist, &object, &pyBGR, &timestamp) )
		return NULL;

	if( !PyArray_Check(object) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "Object passed to cudaFromNumpy() wasn't a numpy ndarray");
		return NULL;
	}

	if( timestamp < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFromNumpy() timestamp cannot be negative");
		return NULL;
	}

	const bool isBGR = (pyBGR > 0);

	// detect uint8 array - otherwise cast to float
	const int inputType = PyArray_TYPE((PyArrayObject*)object);
	int outputType = NPY_FLOAT32;
	int typeSize = sizeof(float);

	if( inputType == NPY_UINT8 )
	{
		outputType = NPY_UINT8;
		typeSize = sizeof(uint8_t);
	}
	
	// cast to numpy array
	PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(object, outputType, NPY_ARRAY_IN_ARRAY|NPY_ARRAY_FORCECAST);

	if( !array )
		return NULL;

	// calculate the size of the array
	const int ndim = PyArray_NDIM(array);
	npy_intp* dims = PyArray_DIMS(array);
	size_t    size = 0;

	for( int n=0; n < ndim; n++ )
	{
		LogDebug(LOG_PY_UTILS "cudaFromNumpy()  ndarray dim %i = %li\n", n, dims[n]);

		if( n == 0 )
			size = dims[0];
		else
			size *= dims[n];
	}

	size *= typeSize;

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

	// detect the image format
	imageFormat format = IMAGE_UNKNOWN;

	if( outputType == NPY_FLOAT32 )
	{
		if( ndim == 2 )
		{
			format = IMAGE_GRAY32F;
		}
		else if( ndim == 3 )
		{
			if( dims[2] == 1 )
				format = IMAGE_GRAY32F;
			else if( dims[2] == 3 )
				format = isBGR ? IMAGE_BGR32F : IMAGE_RGB32F;
			else if( dims[2] == 4 )
				format = isBGR ? IMAGE_BGRA32F : IMAGE_RGBA32F;
		}
	}
	else if( outputType == NPY_UINT8 )
	{
		if( ndim == 2 )
		{
			format = IMAGE_GRAY8;
		}
		else if( ndim == 3 )
		{
			if( dims[2] == 1 )
				format = IMAGE_GRAY8;
			else if( dims[2] == 3 )
				format = isBGR ? IMAGE_BGR8 : IMAGE_RGB8;
			else if( dims[2] == 4 )
				format = isBGR ? IMAGE_BGRA8 : IMAGE_RGBA8;
		}
	}

	// register CUDA memory capsule
	PyObject* capsule = NULL;

	if( format != IMAGE_UNKNOWN )	
		capsule = PyCUDA_RegisterImage(gpuPtr, dims[1], dims[0], format, timestamp, true);
	else
		capsule = PyCUDA_RegisterMemory(gpuPtr, size, true);

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
	{ "cudaFromNumpy", (PyCFunction)PyNumpy_ToCUDA, METH_VARARGS|METH_KEYWORDS, "Copy a numpy ndarray to CUDA memory" },
	{ "cudaToNumpy", (PyCFunction)PyNumpy_FromCUDA, METH_VARARGS|METH_KEYWORDS, "Create a numpy ndarray wrapping the CUDA memory, without copying it" },	
	{NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyNumpy_RegisterFunctions()
{
	return pyImageIO_Functions;
}

// Initialize NumPy
PyMODINIT_FUNC PyNumpy_ImportNumpy()
{
	import_array();
	//import_ufunc();	// only needed if using ufunctions
}

// Register types
bool PyNumpy_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;
	
	PyNumpy_ImportNumpy();
	return true;
}

#else

// stub functions
PyMethodDef* PyNumpy_RegisterFunctions()
{
	LogError(LOG_PY_UTILS "compiled without NumPy array conversion support (warning)\n");
	LogError(LOG_PY_UTILS "if you wish to have support for converting NumPy arrays,\n");
	LogError(LOG_PY_UTILS "first run 'sudo apt-get install python-numpy python3-numpy'\n");

	return NULL;
}

// Register types
bool PyNumpy_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;
	
	return true;
}

#endif

