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

#include "PyImageIO.h"
#include "PyCUDA.h"

#include "loadImage.h"


// PyImageIO_Load
PyObject* PyImageIO_Load( PyObject* self, PyObject* args, PyObject* kwds )
{
	const char* filename  = NULL;
	const char* formatStr = "rgb8";
	long long timestamp = 0;
	cudaStream_t stream = 0;

	static char* kwlist[] = {"filename", "format", "timestamp", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "s|sLK", kwlist, &filename, &formatStr, &timestamp, &stream))
		return NULL;

	const imageFormat format = imageFormatFromStr(formatStr);

	if( format == IMAGE_UNKNOWN )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "loadImage() invalid format string");
		return NULL;
	}

	if( timestamp < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "loadImage() timestamp cannot be negative");
		return NULL;
	}

	// load the image
	void* imgPtr = NULL;
	int   width  = 0;
	int   height = 0;
    bool  result = false;
    
	Py_BEGIN_ALLOW_THREADS
	result = loadImage(filename, &imgPtr, &width, &height, format, stream);
	Py_END_ALLOW_THREADS
    
    if( !result )
	{
		PyErr_Format(PyExc_Exception, LOG_PY_UTILS "loadImage() failed to load image %s", filename);
		return NULL;
	}
    
	return PyCUDA_RegisterImage(imgPtr, width, height, format, timestamp, true);
}


// PyImageIO_LoadRGBA
PyObject* PyImageIO_LoadRGBA( PyObject* self, PyObject* args, PyObject* kwds )
{
	const char* filename  = NULL;
	const char* formatStr = "rgba32f";
	long long timestamp = 0;
	cudaStream_t stream = 0;

    static char* kwlist[] = {"filename", "format", "timestamp", "stream", NULL};
    
	if( !PyArg_ParseTupleAndKeywords(args, kwds, "s|sLK", kwlist, &filename, &formatStr, &timestamp, &stream))
		return NULL;
		
	const imageFormat format = imageFormatFromStr(formatStr);

	if( format == IMAGE_UNKNOWN )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "loadImageRGBA() invalid format string");
		return NULL;
	}

	if( timestamp < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "loadImageRGBA() timestamp cannot be negative");
		return NULL;
	}

	// load the image
	void* imgPtr = NULL;
	int   width  = 0;
	int   height = 0;
    bool  result = false;
    
	Py_BEGIN_ALLOW_THREADS
	result = loadImage(filename, &imgPtr, &width, &height, format, stream);
	Py_END_ALLOW_THREADS
		
    if( !result )
	{
		PyErr_Format(PyExc_Exception, LOG_PY_UTILS "loadImageRGBA() failed to load image %s", filename);
		return NULL;
	}
	
	// register memory container
	PyObject* capsule = PyCUDA_RegisterImage(imgPtr, width, height, format, timestamp, true);

	if( !capsule )
		return NULL;

	// create dimension objects
	PyObject* pyWidth  = PYLONG_FROM_LONG(width);
	PyObject* pyHeight = PYLONG_FROM_LONG(height);

	// return tuple
	PyObject* tuple = PyTuple_Pack(3, capsule, pyWidth, pyHeight);

	Py_DECREF(capsule);
	Py_DECREF(pyWidth);
	Py_DECREF(pyHeight);

	return tuple;
}



// PyImageIO_Save
PyObject* PyImageIO_Save( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	const char* filename = NULL;
	PyObject* capsule = NULL;
	int quality = IMAGE_DEFAULT_SAVE_QUALITY;
	cudaStream_t stream = 0;
	
	static char* kwlist[] = {"filename", "image", "quality", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "sO|iK", kwlist, &filename, &capsule, &quality, &stream))
		return NULL;

	// get pointer to image data
	PyCudaImage* img = PyCUDA_GetImage(capsule);

	if( !img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "saveImage() wasn't passed a cudaImage object");
		return NULL;
	}

	if( !img->base.mapped )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "saveImage() needs to be passed a cudaImage that was allocated in mapped/zeroCopy memory");
		return NULL;
	}

	// save the image
    bool result = false;
	Py_BEGIN_ALLOW_THREADS
	result = saveImage(filename, img->base.ptr, img->width, img->height, img->format, quality, stream);
	Py_END_ALLOW_THREADS
    
    if( !result )
    {
        PyErr_Format(PyExc_Exception, LOG_PY_UTILS "saveImage() failed to save %ix%i image to %s", img->width, img->height, filename);
        return NULL;
    }

	Py_RETURN_NONE;
}

// PyImageIO_SaveRGBA
PyObject* PyImageIO_SaveRGBA( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	const char* filename = NULL;
	PyObject* capsule = NULL;

	int width  = 0;
	int height = 0;
	int quality = IMAGE_DEFAULT_SAVE_QUALITY;

	float max_pixel = 255.0f;
    cudaStream_t stream = 0;
    
	static char* kwlist[] = {"filename", "image", "width", "height", "max_pixel", "quality", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "sO|iifi", kwlist, &filename, &capsule, &width, &height, &max_pixel, &quality, &stream))
		return NULL;

	// get pointer to image data
	PyCudaImage* img = PyCUDA_GetImage(capsule);
    bool save_result = false;
    
	if( img != NULL )
	{
		if( !img->base.mapped )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "saveImageRGBA() needs to be passed a cudaImage that was allocated in mapped/zeroCopy memory");
			return NULL;
		}

        Py_BEGIN_ALLOW_THREADS
		save_result = saveImage(filename, img->base.ptr, img->width, img->height, img->format, quality, make_float2(0,max_pixel), true, stream);
		Py_END_ALLOW_THREADS
	}
	else
	{
		PyCudaMemory* mem = PyCUDA_GetMemory(capsule);

		if( !mem )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "saveImageRGBA() wasn't passed a cudaImage or cudaMemory object");
			return NULL;
		}

		if( !mem->mapped )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "saveImageRGBA() needs to be passed a cudaMemory object that was allocated in mapped/zeroCopy memory");
			return NULL;
		}

		Py_BEGIN_ALLOW_THREADS
		save_result = saveImageRGBA(filename, (float4*)mem->ptr, width, height, max_pixel, quality, stream);
		Py_END_ALLOW_THREADS
	}

    if( !save_result )
    {
        PyErr_Format(PyExc_Exception, LOG_PY_UTILS "saveImage() failed to save %ix%i image to %s", width, height, filename);
        return NULL;
    }

	Py_RETURN_NONE;
}


//-------------------------------------------------------------------------------

static PyMethodDef pyImageIO_Functions[] = 
{
	{ "loadImage", (PyCFunction)PyImageIO_Load, METH_VARARGS|METH_KEYWORDS, "Load an image from disk into GPU memory" },
	{ "loadImageRGBA", (PyCFunction)PyImageIO_LoadRGBA, METH_VARARGS|METH_KEYWORDS, "Load an image from disk into GPU memory as float4 RGBA" },
	{ "saveImage", (PyCFunction)PyImageIO_Save, METH_VARARGS|METH_KEYWORDS, "Save an image to disk" },		
	{ "saveImageRGBA", (PyCFunction)PyImageIO_SaveRGBA, METH_VARARGS|METH_KEYWORDS, "Save a float4 RGBA image to disk" },	
	{NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyImageIO_RegisterFunctions()
{
	return pyImageIO_Functions;
}

// Register types
bool PyImageIO_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;
	
	return true;
}

