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

#include "PyGst.h"
#include "PyCUDA.h"

#include "cudaMappedMemory.h"
#include "cudaYUV.h"

#include <pygobject.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#ifdef HAS_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif // HAS_NUMPY

// cudaFromGstSample()
PyObject* PyGst_SampleToCUDA( PyObject* self, PyObject* args )
{
        PyGObject* pybuf;

        if ( !PyArg_ParseTuple(args, "O", &pybuf) )
        {
                PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "argument should be an object");
                return NULL;
        }

        GstSample* sample = GST_SAMPLE(pybuf->obj);

        // get video info
        GstCaps* caps = gst_sample_get_caps(sample);

        GstVideoInfo video_info;
        gst_video_info_init(&video_info);

        if ( !gst_video_info_from_caps(&video_info, caps) )
        {
                PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gst_video_info_from_caps() failed");
                return NULL;
        }

        if ( g_strcmp0(GST_VIDEO_INFO_NAME(&video_info), "NV12") )
        {
                PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "wrong frame format (NV12 required)");
                return NULL;
        }

        size_t width = video_info.width;
        size_t height = video_info.height;

        // register incoming CUDA memory pointer
        GstBuffer* buffer = gst_sample_get_buffer(sample);

        GstMapInfo info;
        if ( !gst_buffer_map(buffer, &info, GST_MAP_READ) )
        {
                PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gst_buffer_map() failed");
                return NULL;
        }

        // allocate CUDA NV12 memory
        void* nv12GPUPtr = NULL;

        if( CUDA_FAILED(cudaMalloc(&nv12GPUPtr, info.size)) )
        {
                gst_buffer_unmap(buffer, &info);
                PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() NV12 failed");
                return NULL;
        }

        if( CUDA_FAILED(cudaMemcpy(nv12GPUPtr, info.data, info.size, cudaMemcpyHostToDevice)) )
        {
                gst_buffer_unmap(buffer, &info);
                CUDA(cudaFree(nv12GPUPtr));
                PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMemcpy() failed");
                return NULL;
        }

        gst_buffer_unmap(buffer, &info);

        // allocate CUDA RGBA memory and convert from NV12
        void* rgb32CPUPtr = NULL;
        void* rgb32GPUPtr = NULL;

        const size_t size = width * height * sizeof(float4);

        if( !cudaAllocMapped(&rgb32CPUPtr, &rgb32GPUPtr, size) )
        {
                CUDA(cudaFree(nv12GPUPtr));
                PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() RGBA failed");
                return NULL;
        }

        if( CUDA_FAILED(cudaNV12ToRGBA32((uint8_t*)nv12GPUPtr, (float4*)rgb32GPUPtr, width, height)) )
        {
                CUDA(cudaFree(nv12GPUPtr));
                CUDA(cudaFreeHost(rgb32CPUPtr));
                PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaNV12ToRGBA32() failed");
                return NULL;
        }

        // free RGB memory
        CUDA(cudaFree(nv12GPUPtr));

        PyObject* capsule = PyCUDA_RegisterMappedMemory(rgb32CPUPtr, rgb32GPUPtr);

        // create dimension objects
        PyObject* pyWidth  = PYLONG_FROM_LONG(video_info.width);
        PyObject* pyHeight = PYLONG_FROM_LONG(video_info.height);

        // return tuple
        PyObject* tuple = PyTuple_Pack(3, capsule, pyWidth, pyHeight);

        Py_DECREF(capsule);
        Py_DECREF(pyWidth);
        Py_DECREF(pyHeight);

        return tuple;
}

#ifdef HAS_NUMPY
// numpyFromGstBuffer()
PyObject* PyGst_BufferToNumPy( PyObject* self, PyObject* args, PyObject* kwds )
{
        PyGObject* pybuf;

	int width  = 0;
	int height = 0;
	int depth  = 0;

	static char* kwlist[] = {"buffer", "width", "height", "depth", NULL};

        if ( !PyArg_ParseTupleAndKeywords(args, kwds, "Oiii", kwlist, &pybuf, &width, &height, &depth) )
        {
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "numpyFromGstBuffer() failed to parse args tuple");
                return NULL;
        }

        GstBuffer* buffer = GST_BUFFER(pybuf->obj);

        GstMapInfo info;

        if ( !gst_buffer_map(buffer, &info, GST_MAP_READ) )
        {
                gst_buffer_unref(buffer);
                PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gst_buffer_map() failed");
                return NULL;
        }

	// create numpy array
	npy_intp dims[] = { height, width, depth };

        PyArrayObject* array = (PyArrayObject *) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, info.data);
	if( !array )
	{
                gst_buffer_unmap(buffer, &info);
                gst_buffer_unref(buffer);
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to create numpy array");
		return NULL;
	}

        gst_buffer_unmap(buffer, &info);

        // numpy array is now tied to the buffer object
        Py_INCREF((PyObject*)pybuf);
        PyArray_SetBaseObject(array, (PyObject*)pybuf);

	// return the numpy array
	return (PyObject*) array;
}

// numpyToGstBuffer()
PyObject* PyGst_BufferFromNumPy( PyObject* self, PyObject* args, PyObject* kwds )
{
	PyObject* object = NULL;

	if( !PyArg_ParseTuple(args, "O", &object) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "numpyToGstBuffer() failed to parse array argument");
		return NULL;
	}

	// cast to numpy array. it will basically increase the refcount since
	// this should already be an NPY_UINT8 contiguous array.
	PyArrayObject* array = (PyArrayObject*) PyArray_FROM_OTF(object, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
	if( !array )
		return NULL;

	// calculate the size of the array
	const int ndim = PyArray_NDIM(array);

	if( ndim != 3 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "numpyToGstBuffer() numpy ndim is not 3");
		return NULL;
	}

        npy_intp* dims = PyArray_DIMS(array);
	size_t    size = 0;

	for( int n=0; n < ndim; n++ )
	{
		printf(LOG_PY_UTILS "numpyToGstBuffer() ndarray dim %i = %li\n", n, dims[n]);

		if( n == 0 )
			size = dims[0];
		else
			size *= dims[n];


	}

	if( size == 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "numpyToGstBuffer() numpy ndarray has data size of 0 bytes");
		return NULL;
	}

	// retrieve the data pointer to the array
	uint8_t* arrayPtr = (uint8_t*)PyArray_DATA(array);

        // new GstBuffer memory
        gpointer data = g_malloc(size);
        if ( !data )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "numpyToGstBuffer() malloc failed");
		return NULL;
	}

	memcpy(data, arrayPtr, size);

        GstBuffer* buffer = gst_buffer_new_wrapped(data, size);

	// create GstBuffer boxed object
	PyObject* gobj = pyg_boxed_new(GST_TYPE_BUFFER, buffer, false, true);

        Py_DECREF((PyObject*)array);

        return gobj;
}
#endif // HAS_NUMPY

//-------------------------------------------------------------------------------

static PyMethodDef pyGst_Functions[] =
{
        { "cudaFromGstSample", (PyCFunction)PyGst_SampleToCUDA, METH_VARARGS, "Convert an NV12 GstSample to CUDA memory" },
#ifdef HAS_NUMPY
        { "numpyFromGstBuffer", (PyCFunction)PyGst_BufferToNumPy, METH_VARARGS|METH_KEYWORDS, "Convert an RGBA8 GstBuffer to a numpy array" },
        { "numpyToGstBuffer", (PyCFunction)PyGst_BufferFromNumPy, METH_VARARGS, "Convert a numpy array to an RGBA8 GstBuffer" },
#endif // HAS_NUMPY
        {NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyGst_RegisterFunctions()
{
        return pyGst_Functions;
}

#ifdef HAS_NUMPY
// Initialize NumPy
PyMODINIT_FUNC PyGst_ImportNumPy()
{
	import_array();
}
#endif // HAS_NUMPY

// Register types
bool PyGst_RegisterTypes( PyObject* module )
{
        if( !module )
                return false;

        gst_init(NULL, NULL);
        pygobject_init(-1, -1, -1);
#ifdef HAS_NUMPY
	PyGst_ImportNumPy();
#endif // HAS_NUMPY
        return true;
}
