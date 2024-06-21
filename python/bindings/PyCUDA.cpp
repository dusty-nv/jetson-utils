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
#include "cudaColorspace.h"
#include "cudaNormalize.h"
#include "cudaOverlay.h"
#include "cudaResize.h"
#include "cudaCrop.h"
#include "cudaFont.h"
#include "cudaDraw.h"

#include "logging.h"


//-------------------------------------------------------------------------------
// PyCudaMemory_New
static PyObject* PyCudaMemory_New( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyCudaMemory_New()\n");
	
	// allocate a new container
	PyCudaMemory* self = (PyCudaMemory*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_UTILS "cudaMemory tp_alloc() failed to allocate a new object");
		return NULL;
	}
	
	self->ptr = NULL;
	self->size = 0;
	self->mapped = false;
	self->freeOnDelete = true;
    self->stream = NULL;
    self->event = NULL;
    
	return (PyObject*)self;
}

// PyCudaMemory_Dealloc
static void PyCudaMemory_Dealloc( PyCudaMemory* self )
{
	//LogDebug(LOG_PY_UTILS "PyCudaMemory_Dealloc()\n");
	
	Py_BEGIN_ALLOW_THREADS

	self->stream = NULL;
	
	if( self->event )
	{
	    CUDA(cudaEventDestroy(self->event));
	    self->event = NULL;
	}
	   
	if( self->freeOnDelete && self->ptr != NULL )
	{
		if( self->mapped )
			CUDA(cudaFreeHost(self->ptr));
		else
			CUDA(cudaFree(self->ptr));

		self->ptr = NULL;
	}
	
	Py_END_ALLOW_THREADS

	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}

// PyCudaMemory_Init
static int PyCudaMemory_Init( PyCudaMemory* self, PyObject *args, PyObject *kwds )
{
	//LogDebug(LOG_PY_UTILS "PyCudaMemory_Init()\n");
	
	// parse arguments
	int size = 0;
	int mapped = 1;
	int freeOnDelete = 1;

	static char* kwlist[] = {"size", "mapped", "freeOnDelete", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "i|ii", kwlist, &size, &mapped, &freeOnDelete))
		return -1;
    
	if( size < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMemory.__init()__ had invalid size");
		return -1;
	}

	// allocate CUDA memory
	if( mapped > 0 )
	{
	    PYCUDA_CHECK_NOGIL(cudaMallocMapped(&self->ptr, size), -1);
	}
	else
	{
		PYCUDA_CHECK_NOGIL(cudaMalloc(&self->ptr, size), -1);
	}

	self->size = size;
	self->mapped = (mapped > 0) ? true : false;
	self->freeOnDelete = (freeOnDelete > 0) ? true : false;

	return 0;
}

// PyCudaMemory_ToString
static PyObject* PyCudaMemory_ToString( PyCudaMemory* self )
{
	char str[1024];

    if( self->stream || self->event )
    {
	    sprintf(str, 
		       "<cudaMemory object>\n"
		       "   -- ptr:    %p\n"
		       "   -- size:   %zu\n"
		       "   -- stream: %p\n"
		       "   -- event:  %p\n"
		       "   -- mapped: %s\n"
		       "   -- freeOnDelete: %s\n",
		       self->ptr, self->size, 
		       self->stream, self->event,
		       self->mapped ? "true" : "false", 
		       self->freeOnDelete ? "true" : "false");
    }
    else
    {
        sprintf(str, 
		       "<cudaMemory object>\n"
		       "   -- ptr:    %p\n"
		       "   -- size:   %zu\n"
		       "   -- mapped: %s\n"
		       "   -- freeOnDelete: %s\n",
		       self->ptr, self->size, 
		       self->mapped ? "true" : "false", 
		       self->freeOnDelete ? "true" : "false");
    }
    
	return PYSTRING_FROM_STRING(str);
}

// PyCudaMemory_GetPtr
static PyObject* PyCudaMemory_GetPtr( PyCudaMemory* self, void* closure )
{
	return PYLONG_FROM_UNSIGNED_LONG((uint64_t)self->ptr);
}

// PyCudaMemory_GetSize
static PyObject* PyCudaMemory_GetSize( PyCudaMemory* self, void* closure )
{
	return PYLONG_FROM_UNSIGNED_LONG(self->size);
}

// PyCudaMemory_GetMapped
static PyObject* PyCudaMemory_GetMapped( PyCudaMemory* self, void* closure )
{
	PY_RETURN_BOOL(self->mapped);
}

// PyCudaMemory_GetFreeOnDelete
static PyObject* PyCudaMemory_GetFreeOnDelete( PyCudaMemory* self, void* closure )
{
	PY_RETURN_BOOL(self->freeOnDelete);
}

// PyCudaMemory_GetStream
static PyObject* PyCudaMemory_GetStream( PyCudaMemory* self, void* closure )
{
	return PYLONG_FROM_PTR(self->stream);
}

// PyCudaMemory_SetStream
static int PyCudaMemory_SetStream( PyCudaMemory* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "Not permitted to delete cudaMemory.stream attribute");
		return -1;
	}

    cudaStream_t stream = NULL;
    
    if( value != Py_None )
    {
        stream = (cudaStream_t)PyLong_AsUnsignedLongLong(value);

	    if( PyErr_Occurred() != NULL )
		    return -1;
    }

    self->stream = stream;
	return 0;
}

// PyCudaMemory_GetEvent
static PyObject* PyCudaMemory_GetEvent( PyCudaMemory* self, void* closure )
{
	return PYLONG_FROM_PTR(self->event);
}

// PyCudaMemory_GetEvent
static int PyCudaMemory_SetEvent( PyCudaMemory* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "Not permitted to delete cudaMemory.event attribute");
		return -1;
	}

    cudaEvent_t event = NULL;
    
    if( value != Py_None )
    {
        event = (cudaEvent_t)PyLong_AsUnsignedLongLong(value);

	    if( PyErr_Occurred() != NULL )
		    return -1;
    }
    
    if( self->event != NULL )
        CUDA(cudaEventDestroy(self->event));

    self->event = event;
	return 0;
}

static PyGetSetDef pyCudaMemory_GetSet[] = 
{
	{ "ptr", (getter)PyCudaMemory_GetPtr, NULL, "Address of CUDA memory", NULL},
	{ "size", (getter)PyCudaMemory_GetSize, NULL, "Size (in bytes)", NULL},
	{ "mapped", (getter)PyCudaMemory_GetMapped, NULL, "Is the memory mapped to CPU also? (zeroCopy)", NULL},
	{ "freeOnDelete", (getter)PyCudaMemory_GetFreeOnDelete, NULL, "Will the CUDA memory be released when the Python object is deleted?", NULL},	
	{ "stream", (getter)PyCudaMemory_GetStream, (setter)PyCudaMemory_SetStream, "The CUDA stream last associated with this memory", NULL},	
	{ "event", (getter)PyCudaMemory_GetEvent, (setter)PyCudaMemory_SetEvent, "The CUDA event last associated with this memory", NULL},
	{ "gpudata", (getter)PyCudaMemory_GetPtr, NULL, "Address of CUDA memory (PyCUDA interface)", NULL},
	{ NULL } /* Sentinel */
};

static PyTypeObject pyCudaMemory_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

// PyCudaMemory_RegisterType
bool PyCudaMemory_RegisterType( PyObject* module )
{
	if( !module )
		return false;
	
	pyCudaMemory_Type.tp_name 	  = PY_UTILS_MODULE_NAME ".cudaMemory";
	pyCudaMemory_Type.tp_basicsize  = sizeof(PyCudaMemory);
	pyCudaMemory_Type.tp_flags 	  = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyCudaMemory_Type.tp_methods  = NULL;
	pyCudaMemory_Type.tp_getset   = pyCudaMemory_GetSet;
	pyCudaMemory_Type.tp_new 	  = PyCudaMemory_New;
	pyCudaMemory_Type.tp_init     = (initproc)PyCudaMemory_Init;
	pyCudaMemory_Type.tp_dealloc  = (destructor)PyCudaMemory_Dealloc;
	pyCudaMemory_Type.tp_str      = (reprfunc)PyCudaMemory_ToString;
	pyCudaMemory_Type.tp_doc  	  = "CUDA memory";
	 
	if( PyType_Ready(&pyCudaMemory_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "PyCudaMemory PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyCudaMemory_Type);
    
	if( PyModule_AddObject(module, "cudaMemory", (PyObject*)&pyCudaMemory_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "PyCudaMemory PyModule_AddObject('cudaMemory') failed\n");
		return false;
	}
	
	return true;
}

//-------------------------------------------------------------------------------
// PyCudaImage_New
static PyObject* PyCudaImage_New( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyCudaImage_New()\n");
	
	// allocate a new container
	PyCudaImage* self = (PyCudaImage*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_UTILS "cudaImage tp_alloc() failed to allocate a new object");
		return NULL;
	}
	
	self->base.ptr = NULL;
	self->base.size = 0;
	self->base.mapped = false;
	self->base.freeOnDelete = true;
	self->base.stream = NULL;
    self->base.event = NULL;
    
	self->width = 0;
	self->height = 0;
	
	self->shape[0] = 0; self->shape[1] = 0; self->shape[2] = 0;
	self->strides[0] = 0; self->strides[1] = 0; self->strides[2] = 0;
	
	self->format = IMAGE_UNKNOWN;
	self->timestamp = 0;
	self->cudaArrayInterfaceDict = NULL;
	
	return (PyObject*)self;
}

// PyCudaImage_Config
static void PyCudaImage_Config( PyCudaImage* self, void* ptr, uint32_t width, uint32_t height, imageFormat format, uint64_t timestamp, bool mapped, bool freeOnDelete )
{
	self->base.ptr = ptr;
	self->base.size = imageFormatSize(format, width, height);
	self->base.mapped = mapped;
	self->base.freeOnDelete = freeOnDelete;
    self->base.stream = NULL;
    self->base.event = NULL;
    
	const size_t bitDepth = imageFormatDepth(format);
	
	self->width = width;
	self->height = height;
	
	self->shape[0] = height;
	self->shape[1] = width;
	self->shape[2] = imageFormatChannels(format);

	self->strides[0] = (width * bitDepth) / 8;
	self->strides[1] = bitDepth / 8;
	self->strides[2] = self->strides[1] / self->shape[2];

	self->format = format;
	self->timestamp = timestamp;
	self->cudaArrayInterfaceDict = NULL;
}

// PyCudaImage_Init
static int PyCudaImage_Init( PyCudaImage* self, PyObject *args, PyObject *kwds )
{
	//LogDebug(LOG_PY_UTILS "PyCudaImage_Init()\n");
	
	// parse arguments
	int width = 0;
	int height = 0;
	int mapped = -1;
	int freeOnDelete = -1;
	
	long long timestamp = 0;
	long long externPtr = 0;
	
	PyObject* pyImageLike = NULL;
	
	const char* formatStr = "rgb8";
	static char* kwlist[] = {"width", "height", "format", "timestamp", "mapped", "freeOnDelete", "ptr", "like", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "ii|sLpiLO", kwlist, &width, &height, &formatStr, &timestamp, &mapped, &freeOnDelete, &externPtr, &pyImageLike))
		return -1;
	
	imageFormat format = imageFormatFromStr(formatStr);

	if( format == IMAGE_UNKNOWN )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaImage.__init()__ had invalid image format");
		return -1;
	}
	
	if( pyImageLike != NULL )
	{
		PyCudaImage* image_like = PyCUDA_GetImage(pyImageLike);
		
		width = image_like->width;
		height = image_like->height;
		format = image_like->format;
	}
	
	if( width < 0 || height < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaImage.__init()__ had invalid width/height");
		return -1;
	}

	if( timestamp < 0)
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaImage.__init()__ had invalid timestamp");
		return -1;
	}

	const size_t size = imageFormatSize(format, width, height);
	
	if( externPtr != 0 )
	{
		// import external memory
		self->base.ptr = (void*)externPtr;
		
		if( mapped < 0 )
			mapped = 0;
		
		if( freeOnDelete < 0 )
			freeOnDelete = 0;
	}
	else 
	{
		// allocate CUDA memory
		if( mapped < 0 )
			mapped = 1;
		
		if( freeOnDelete < 0 )
			freeOnDelete = 1;
		
	    if( mapped > 0 )
	    {
	        PYCUDA_CHECK_NOGIL(cudaMallocMapped(&self->base.ptr, size), -1);
	    }
	    else
	    {
		    PYCUDA_CHECK_NOGIL(cudaMalloc(&self->base.ptr, size), -1);
	    }
	}

	PyCudaImage_Config(self, self->base.ptr, width, height, format, timestamp, (mapped > 0) ? true : false, (freeOnDelete > 0) ? true : false);
	return 0;
}

// PyCudaImage_ToString
static PyObject* PyCudaImage_ToString( PyCudaImage* self )
{
	char str[1024];

    if( self->base.stream || self->base.event )
    {
	    sprintf(str, 
		       "<cudaImage object>\n"
		       "   -- ptr:      %p\n"
		       "   -- size:     %zu\n"
		       "   -- width:    %u\n"
		       "   -- height:   %u\n"
		       "   -- channels: %u\n"
		       "   -- format:   %s\n"
		       "   -- stream:   %p\n"
		       "   -- event:    %p\n"
		       "   -- mapped:   %s\n"
		       "   -- freeOnDelete: %s\n"
		       "   -- timestamp:    %f\n",
		       self->base.ptr, self->base.size, (uint32_t)self->width, (uint32_t)self->height, (uint32_t)self->shape[2],  
		       imageFormatToStr(self->format), self->base.stream, self->base.event, self->base.mapped ? "true" : "false", self->base.freeOnDelete ? "true" : "false",
		       self->timestamp / 1.0e+9);
    }
    else
    {
	    sprintf(str, 
	       "<cudaImage object>\n"
	       "   -- ptr:      %p\n"
	       "   -- size:     %zu\n"
	       "   -- width:    %u\n"
	       "   -- height:   %u\n"
	       "   -- channels: %u\n"
	       "   -- format:   %s\n"
	       "   -- mapped:   %s\n"
	       "   -- freeOnDelete: %s\n"
	       "   -- timestamp:    %f\n",
	       self->base.ptr, self->base.size, (uint32_t)self->width, (uint32_t)self->height, (uint32_t)self->shape[2],  
	       imageFormatToStr(self->format), self->base.mapped ? "true" : "false", self->base.freeOnDelete ? "true" : "false",
	       self->timestamp / 1.0e+9);
    }
    
	return PYSTRING_FROM_STRING(str);
}

// PyCudaImage_GetWidth
static PyObject* PyCudaImage_GetWidth( PyCudaImage* self, void* closure )
{
	return PYLONG_FROM_UNSIGNED_LONG(self->width);
}

// PyCudaImage_GetHeight
static PyObject* PyCudaImage_GetHeight( PyCudaImage* self, void* closure )
{
	return PYLONG_FROM_UNSIGNED_LONG(self->height);
}

// PyCudaImage_GetChannels
static PyObject* PyCudaImage_GetChannels( PyCudaImage* self, void* closure )
{
	return PYLONG_FROM_UNSIGNED_LONG(self->shape[2]);
}

// PyCudaImage_GetShape
static PyObject* PyCudaImage_GetShape( PyCudaImage* self, void* closure )
{
	PyObject* height   = PYLONG_FROM_UNSIGNED_LONG(self->shape[0]);
	PyObject* width    = PYLONG_FROM_UNSIGNED_LONG(self->shape[1]);
	PyObject* channels = PYLONG_FROM_UNSIGNED_LONG(self->shape[2]);

	PyObject* tuple = PyTuple_Pack(3, height, width, channels);

	Py_DECREF(height);
	Py_DECREF(width);
	Py_DECREF(channels);

	return tuple;
}

// PyCudaImage_GetFormat
static PyObject* PyCudaImage_GetFormat( PyCudaImage* self, void* closure )
{
	return PYSTRING_FROM_STRING(imageFormatToStr(self->format));
}

// PyCudaImage_GetTimestamp
static PyObject* PyCudaImage_GetTimestamp( PyCudaImage* self, void* closure )
{
	return PYLONG_FROM_UNSIGNED_LONG_LONG(self->timestamp);
}


// imageFormatToNumpyTypeStr
static const char* imageFormatToNumpyTypeStr( imageFormat format )
{
	// https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
	const imageBaseType baseType = imageFormatBaseType(format);

	if( baseType == IMAGE_FLOAT )
		return "<f4";
	else if( baseType == IMAGE_UINT8 )
		return "<u1";

	return "V";
}

// DICT_SET macro
#define DICT_SET(dict, key, value) 													\
	if( PyDict_SetItemString(dict, key, value) != 0 ) 									\
		return PyErr_Format(PyExc_Exception, LOG_PY_UTILS "failed to set key '%s' in dict", key); \
	Py_DECREF(value)
	
// PyCudaImage_GetCudaArrayInterface
static PyObject* PyCudaImage_GetCudaArrayInterface( PyCudaImage* self, void* closure )
{
	if( self->cudaArrayInterfaceDict != NULL )
	{
		Py_INCREF(self->cudaArrayInterfaceDict);
		return self->cudaArrayInterfaceDict;
	}
	
	//LogDebug(LOG_PY_UTILS "PyCudaImage creating __cuda_array_interface__ dict\n");
	
	// https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
	PyObject* dict = PyDict_New();
	
	PyObject* shape = PyCudaImage_GetShape(self, closure);
	PyObject* typestr = PYSTRING_FROM_STRING(imageFormatToNumpyTypeStr(self->format));
	PyObject* version = PYLONG_FROM_LONG(3);
	
	PyObject* data_ptr = PYLONG_FROM_UNSIGNED_LONG_LONG((uint64_t)self->base.ptr);
	PyObject* data_tuple = PyTuple_Pack(2, data_ptr, Py_False);
	
	Py_DECREF(data_ptr);

	// set dictionary keys
	DICT_SET(dict, "shape", shape);
	DICT_SET(dict, "typestr", typestr);
	DICT_SET(dict, "data", data_tuple);
	DICT_SET(dict, "version", version);
	
	Py_INCREF(dict);
	
	self->cudaArrayInterfaceDict = dict;
	return self->cudaArrayInterfaceDict;
}

// PyCudaImage_GetArrayInterface
static PyObject* PyCudaImage_GetArrayInterface( PyCudaImage* self, void* closure )
{
	if( !self->base.mapped )
	{
		LogDebug(LOG_PY_UTILS "PyCudaImage => returning None for __array_interface__ attribute for unmapped buffer");
		Py_RETURN_NONE;
	}
	
	// https://numpy.org/doc/stable/reference/arrays.interface.html
	// numpy interface is nearly identical to numba interface
	return PyCudaImage_GetCudaArrayInterface(self, closure);
}

// PyCudaImage_ParseSubscriptTuple
static int PyCudaImage_ParseSubscriptTuple(PyCudaImage* self, PyObject* tuple, int* numComponents, bool* exceptionSet)
{
	if( !PyTuple_Check(tuple) )
		return -1;

	// support between 1 and 3 tuple indices:
	//    1. img[y*width+x]
	//    2. img[y,x]
	//    3. img[y,x,channel]
	const Py_ssize_t tupleSize = PyTuple_Size(tuple);

	if( tupleSize <= 0 || tupleSize > 3 )
		return -1;

	const size_t imgChannels = imageFormatChannels(self->format);
	const size_t dimSize[] = { self->height, self->width, imgChannels };
	int dims[] = {-1, -1, -1};

	for( int n=0; n < tupleSize; n++ )
	{
		const long dim = PYLONG_AS_LONG(PyTuple_GetItem(tuple, n));

		if( dim == -1 && PyErr_Occurred() != NULL )
		{
			PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "cudaImage subscript had invalid element in key tuple");
			*exceptionSet = true;
			return -1;
		}
		
		dims[n] = dim;

		// wrap around negative indices
		if( dims[n] < 0 )
			dims[n] += dimSize[n];

		// confirm the dim is in-range
		if( dims[n] < 0 || dims[n] >= dimSize[n] )
		{
			PyErr_SetString(PyExc_IndexError, LOG_PY_UTILS "cudaImage subscript was out of range");
			*exceptionSet = true;
			return -1;
		}
	}
	
	const size_t pixelDepth = imageFormatDepth(self->format);
	const size_t baseDepth = pixelDepth / imgChannels;
	
	if( tupleSize == 1 )
	{
		// pixel index - img[y * img.width + x]
		*numComponents = imageFormatChannels(self->format);
		return (dims[0] * pixelDepth) / 8;
	}
	else if( tupleSize == 2 )
	{
		// y, x index - img[y,x]
		*numComponents = imageFormatChannels(self->format);
		return ((dims[0] * self->width + dims[1]) * pixelDepth) / 8;
	}
	else if( tupleSize == 3 )
	{
		// individual component index - img[y,x,channel]
		*numComponents = 1;
		return (((dims[0] * self->width + dims[1]) * pixelDepth) + dims[2] * baseDepth) / 8;	// return byte offset
	}

	return -1;
}

// PyCudaImage_ParseSubscriptOffset
static int PyCudaImage_ParseSubscript(PyCudaImage* self, PyObject* key, int* numComponents)
{
	//PyObject_Print(PyObject_Type(key), stdout, Py_PRINT_RAW);
	int offset = PYLONG_AS_LONG(key);

	if( offset == -1 && PyErr_Occurred() != NULL )
	{
		PyErr_Clear();
		bool exceptionSet = false;
		offset = PyCudaImage_ParseSubscriptTuple(self, key, numComponents, &exceptionSet);

		if( offset < 0 )
		{
			if( !exceptionSet )
				PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "cudaImage subscript had invalid key");

			return -1;
		}

		return offset;
	}

	// negative indexes wrap around
	if( offset < 0 )
		offset = self->width * self->height + offset;

	// bounds checking
	if( offset < 0 || offset >= (self->width * self->height) )
	{
		PyErr_SetString(PyExc_IndexError, LOG_PY_UTILS "cudaImage subscript was out of range");
		return -1;
	}

	*numComponents = imageFormatChannels(self->format);
	offset = (offset * imageFormatDepth(self->format)) / 8;
	return offset;
}

// PyCudaImage_GetItem
static PyObject* PyCudaImage_GetItem(PyCudaImage *self, PyObject *key)
{
	if( !self->base.mapped )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaImage subscript operator can only operate on mapped/zeroCopy memory");
		return NULL;
	}

	int numComponents = 0;
	const int offset = PyCudaImage_ParseSubscript(self, key, &numComponents);

	if( offset < 0 )
		return NULL;
	
	// apply offset to the data pointer
	uint8_t* ptr = ((uint8_t*)self->base.ptr) + offset;
	const imageBaseType baseType = imageFormatBaseType(self->format);
	
	if( numComponents > 1 )
	{
		// return the pixel as a tuple
		PyObject* tuple = PyTuple_New(numComponents);
		
		for( int n=0; n < numComponents; n++ )
		{
			PyObject* component = NULL;

			if( baseType == IMAGE_FLOAT )
				component = PyFloat_FromDouble(((float*)ptr)[n]);
			else if( baseType == IMAGE_UINT8 )
				component = PYLONG_FROM_UNSIGNED_LONG(ptr[n]);
			
			PyTuple_SetItem(tuple, n, component);
		}
		
		return tuple;
	}
	else
	{
		if( baseType == IMAGE_FLOAT )
			return PyFloat_FromDouble(((float*)ptr)[0]);
		else if( baseType == IMAGE_UINT8 )
			return PYLONG_FROM_UNSIGNED_LONG(ptr[0]);
		else
			return NULL;  // suppress compiler return warning
	}
}

// PyCudaImage_SetItem
static int PyCudaImage_SetItem( PyCudaImage* self, PyObject* key, PyObject* value )
{
	if( !self->base.mapped )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaImage subscript operator can only operate on mapped/zeroCopy memory");
		return -1;
	}

	int numComponents = 0;
	const int offset = PyCudaImage_ParseSubscript(self, key, &numComponents);

	if( offset < 0 )
		return -1;
	
	// apply offset to the data pointer
	uint8_t* ptr = ((uint8_t*)self->base.ptr) + offset;
	const size_t imgChannels = imageFormatChannels(self->format);
	const imageBaseType baseType = imageFormatBaseType(self->format);
	
	// if this is a list, convert it to tuple
	PyObject* list = NULL;
	
	if( PyList_Check(value) )
	{
		list = PyList_AsTuple(value);
		value = list;
	}
	
	// macro to parse object and assign it to a channel
	#define assign_pixel_channel(channel, value)    \
	{												\
		const float val = PyFloat_AsDouble(value);  \
													\
		if( PyErr_Occurred() != NULL )				\
		{											\
			PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "cudaImage subscript was assigned an invalid value (int or float expected)"); \
			return -1; 								\
		} 											\
													\
		if( baseType == IMAGE_FLOAT )				\
			((float*)ptr)[channel] = val;			\
		else if( baseType == IMAGE_UINT8 )			\
			ptr[channel] = val;						\
	}
	
	// check if this is a tuple
	if( !PyTuple_Check(value) )
	{
		// if an individual channel is being set, try a single float/int
		if( numComponents == 1 )
		{
			assign_pixel_channel(0, value);
			return 0;
		}

		PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "cudaImage subscript was assigned an invalid value (tuple or list expected)");
		return -1;
	}
	
	// check that the tuple length matches the number of image channels
	const Py_ssize_t tupleSize = PyTuple_Size(value);

	if( tupleSize != numComponents )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "cudaImage subscript was assigned a tuple with a different length than the number of image channels");
		return -1;
	}

	// assign each tuple element to a channel
	for( int n=0; n < tupleSize; n++ )
	{
		PyObject* tupleItem = PyTuple_GetItem(value, n);
		assign_pixel_channel(n, tupleItem);
	}

	if( list != NULL )
		Py_DECREF(list);
	
	return 0;
}

// PyCudaImage_Length
static Py_ssize_t PyCudaImage_Length(PyCudaImage* self) 
{
	return self->width * self->height;
}

static PyMappingMethods pyCudaImage_AsMapping = {
	(lenfunc)PyCudaImage_Length,
	(binaryfunc)PyCudaImage_GetItem,
	(objobjargproc)PyCudaImage_SetItem,
};

#if PY_MAJOR_VERSION >= 3
// PyCudaImage_GetBuffer
static int PyCudaImage_GetBuffer(PyCudaImage* self, Py_buffer* view, int flags)
{
	if( view == NULL ) 
	{
		PyErr_SetString(PyExc_BufferError, "cudaImage - NULL view in buffer view");
		return -1;
	}

	if( !self->base.mapped )
	{
		PyErr_SetString(PyExc_BufferError, "cudaImage - buffer must be allocated as mapped to view as buffer");
		return -1;
	}	
	
	view->obj = (PyObject*)self;
	view->buf = (void*)self->base.ptr;
	view->len = self->base.size;
	view->readonly = 0;
	view->itemsize = (imageFormatDepth(self->format) / 8) / imageFormatChannels(self->format);
	
	view->ndim = 3; //(self->shape[2] > 1) ? 3 : 2;
	view->shape = self->shape;  // length-1 sequence of dimensions
	view->strides = self->strides;  // for the simple case we can do this
	view->suboffsets = NULL;
	view->internal = NULL;

	switch(self->format)
	{
		case IMAGE_RGB8:
		case IMAGE_BGR8:		
		case IMAGE_RGBA8:		
		case IMAGE_BGRA8:		
		case IMAGE_GRAY8:				
		case IMAGE_I420:
		case IMAGE_YV12:
		case IMAGE_NV12:		
		case IMAGE_UYVY:
		case IMAGE_YUYV:		
		case IMAGE_BAYER_BGGR:
		case IMAGE_BAYER_GBRG:
		case IMAGE_BAYER_GRBG:
		case IMAGE_BAYER_RGGB:	view->format = "B";	break;
		case IMAGE_RGB32F:		
		case IMAGE_RGBA32F: 	
		case IMAGE_GRAY32F:		view->format = "f"; break;
	}
	
	Py_INCREF(self);
	return 0;
}

static PyBufferProcs pyCudaImage_AsBuffer = {
	// this definition is only compatible with Python 3.3 and above
	(getbufferproc)PyCudaImage_GetBuffer,
	(releasebufferproc)0,  // we do not require any special release function
};

#endif

static PyGetSetDef pyCudaImage_GetSet[] = 
{
	{ "width", (getter)PyCudaImage_GetWidth, NULL, "Width of the image (in pixels)", NULL},
	{ "height", (getter)PyCudaImage_GetHeight, NULL, "Height of the image (in pixels)", NULL},
	{ "channels", (getter)PyCudaImage_GetChannels, NULL, "Number of color channels in the image", NULL},
	{ "shape", (getter)PyCudaImage_GetShape, NULL, "Image dimensions in (height, width, channels) tuple", NULL},
	{ "format", (getter)PyCudaImage_GetFormat, NULL, "Pixel format of the image", NULL},
	{ "timestamp", (getter)PyCudaImage_GetTimestamp, NULL, "Timestamp of the image (in nanoseconds)", NULL},
	{ "__array_interface__", (getter)PyCudaImage_GetArrayInterface, NULL, "Numpy __array_interface__ dict", NULL},
	{ "__cuda_array_interface__", (getter)PyCudaImage_GetCudaArrayInterface, NULL, "Numba __cuda_array_interface__ dict", NULL},
	{ NULL } /* Sentinel */
};

static PyTypeObject pyCudaImage_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

// PyCudaImage_RegisterType
bool PyCudaImage_RegisterType( PyObject* module )
{
	if( !module )
		return false;
	
	pyCudaImage_Type.tp_name 	= PY_UTILS_MODULE_NAME ".cudaImage";
	pyCudaImage_Type.tp_basicsize = sizeof(PyCudaImage);
	pyCudaImage_Type.tp_flags 	= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyCudaImage_Type.tp_base    = &pyCudaMemory_Type;
	pyCudaImage_Type.tp_methods = NULL;
	pyCudaImage_Type.tp_getset  = pyCudaImage_GetSet;
	pyCudaImage_Type.tp_as_mapping = &pyCudaImage_AsMapping;
	pyCudaImage_Type.tp_new     = PyCudaImage_New;
	pyCudaImage_Type.tp_init    = (initproc)PyCudaImage_Init;
	pyCudaImage_Type.tp_dealloc	= NULL; /*(destructor)PyCudaMemory_Dealloc*/;
	pyCudaImage_Type.tp_str		= (reprfunc)PyCudaImage_ToString;
	pyCudaImage_Type.tp_doc  	= "CUDA image";
	
#if PY_MAJOR_VERSION >= 3
	pyCudaImage_Type.tp_as_buffer  = &pyCudaImage_AsBuffer;
#endif

	if( PyType_Ready(&pyCudaImage_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "PyCudaImage PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyCudaImage_Type);
    
	if( PyModule_AddObject(module, "cudaImage", (PyObject*)&pyCudaImage_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "PyCudaImage PyModule_AddObject('cudaImage') failed\n");
		return false;
	}
	
	return true;
}

//-------------------------------------------------------------------------------
// PyCUDA_RegisterMemory
PyObject* PyCUDA_RegisterMemory( void* gpuPtr, size_t size, bool mapped, bool freeOnDelete )
{
	if( !gpuPtr )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "RegisterMemory() was provided NULL memory pointers");
		return NULL;
	}

	PyCudaMemory* mem = PyObject_New(PyCudaMemory, &pyCudaMemory_Type);

	if( !mem )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "PyCUDA_RegisterMemory() failed to create a new cudaMemory object");
		return NULL;
	}

	mem->ptr = gpuPtr;
	mem->size = size;
	mem->mapped = mapped;
	mem->freeOnDelete = freeOnDelete;
    mem->stream = NULL;
    mem->event = NULL;
    
	return (PyObject*)mem;
}

// PyCUDA_RegisterImage
PyObject* PyCUDA_RegisterImage( void* gpuPtr, uint32_t width, uint32_t height, imageFormat format, uint64_t timestamp, bool mapped, bool freeOnDelete )
{
	if( !gpuPtr )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaRegisterImage() was provided NULL memory pointers");
		return NULL;
	}

	PyCudaImage* mem = PyObject_New(PyCudaImage, &pyCudaImage_Type);

	if( !mem )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "PyCUDA_RegisterImage() failed to create a new cudaImage object");
		return NULL;
	}

	PyCudaImage_Config(mem, gpuPtr, width, height, format, timestamp, mapped, freeOnDelete);
	return (PyObject*)mem;
}

// PyCUDA_IsMemory
bool PyCUDA_IsMemory( PyObject* object )
{
	if( !object )
		return false;

	if( PyObject_IsInstance(object, (PyObject*)&pyCudaMemory_Type) == 1 || 
	    PyObject_IsInstance(object, (PyObject*)&pyCudaImage_Type) == 1)
	{
		return true;
	}

	return false;
}

// PyCUDA_IsImage
bool PyCUDA_IsImage( PyObject* object )
{
	if( !object )
		return false;

	if( PyObject_IsInstance(object, (PyObject*)&pyCudaImage_Type) == 1 )
		return true;

	return false;
}

// PyCUDA_GetMemory
PyCudaMemory* PyCUDA_GetMemory( PyObject* object )
{
	if( !object )
		return NULL;

	if( PyCUDA_IsMemory(object) )
		return (PyCudaMemory*)object;

	return NULL;
}

// PyCUDA_GetImage
PyCudaImage* PyCUDA_GetImage( PyObject* object )
{
	if( !object )
		return NULL;

	if( PyCUDA_IsImage(object) )
		return (PyCudaImage*)object;

	return NULL;
}

// PyCUDA_GetImage
void* PyCUDA_GetImage( PyObject* capsule, int* width, int* height, imageFormat* format, uint64_t* timestamp )
{
	PyCudaImage* img = PyCUDA_GetImage(capsule);
	void* ptr = NULL;

	if( img != NULL )
	{
		ptr = img->base.ptr;
		*width = img->width;
		*height = img->height;
		*format = img->format;
		if ( timestamp )
			*timestamp = img->timestamp;
	}
	else
	{
		PyCudaMemory* mem = PyCUDA_GetMemory(capsule);

		if( !mem )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "function wasn't passed a valid cudaImage or cudaMemory object");
			return NULL;
		}

		ptr = mem->ptr;

		if( *width <= 0 || *height <= 0 )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "image dimensions are invalid");
			return NULL;
		}

		if( *format == IMAGE_UNKNOWN )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "invalid image format");
			return NULL;
		}
	}

	if( !ptr )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "image pointer was NULL (should be cudaImage or cudaMemory)");
		return NULL;
	}

	return ptr;
}


//-------------------------------------------------------------------------------
// PyCUDA_Malloc
PyObject* PyCUDA_Malloc( PyObject* self, PyObject* args, PyObject* kwds )
{
	int size = 0;
	int width = 0;
	int height = 0;
	long long timestamp = 0;
    PyObject* pyLike = NULL;
    
	const char* formatStr = NULL;
	static char* kwlist[] = {"size", "width", "height", "format", "timestamp", "like", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|iiisLO", kwlist, &size, &width, &height, &formatStr, &timestamp, &pyLike) )
		return NULL;

    imageFormat format = imageFormatFromStr(formatStr);
    
    if( pyLike != NULL )
	{
	    if( PyCUDA_IsImage(pyLike) )
	    {
	        PyCudaImage* img = (PyCudaImage*)pyLike;
	        
	        size = img->base.size;
	        width = img->width;
	        height = img->height;
	        format = img->format;
	        timestamp = img->timestamp;
	    }
	    else if( PyCUDA_IsMemory(pyLike) )
	    {
	        size = ((PyCudaMemory*)pyLike)->size;
	    }
	    else
	    {
	        PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "the 'like' object should be of type cudaImage or cudaMemory");
	        return NULL;
	    }
	}
	
	if( size <= 0 && (width <= 0 || height <= 0) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() requested size/dimensions are negative or zero");
		return NULL;
	}

	if( timestamp < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() timestamp cannot be negative");
		return NULL;
	}
	
	const bool isImage = (width > 0) && (height > 0);

	if( isImage && format == IMAGE_UNKNOWN )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() invalid format string");
		return NULL;
	}

	if( isImage )
		size = imageFormatSize(format, width, height);

	// allocate memory
	void* ptr = NULL;

    PYCUDA_ASSERT_NOGIL(cudaMalloc(&ptr, size));

	return isImage ? PyCUDA_RegisterImage(ptr, width, height, format, timestamp)
                   : PyCUDA_RegisterMemory(ptr, size);
}


// cudaAllocMapped (cudaMallocMapped)
PyObject* PyCUDA_AllocMapped( PyObject* self, PyObject* args, PyObject* kwds )
{
	int size = 0;
	int clear = 1;
	float width = 0;
	float height = 0;
	long long timestamp = 0;
	PyObject* pyLike = NULL;

	const char* formatStr = NULL;
	static char* kwlist[] = {"size", "width", "height", "format", "timestamp", "like", "clear", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|iffsLOp", kwlist, &size, &width, &height, &formatStr, &timestamp, &pyLike, &clear))
		return NULL;

	imageFormat format = imageFormatFromStr(formatStr);
	
    if( pyLike != NULL )
	{
	    if( PyCUDA_IsImage(pyLike) )
	    {
	        PyCudaImage* img = (PyCudaImage*)pyLike;
	        
	        size = img->base.size;
	        width = img->width;
	        height = img->height;
	        format = img->format;
	        timestamp = img->timestamp;
	    }
	    else if( PyCUDA_IsMemory(pyLike) )
	    {
	        size = ((PyCudaMemory*)pyLike)->size;
	    }
	    else
	    {
	        PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "the 'like' object should be of type cudaImage or cudaMemory");
	        return NULL;
	    }
	}
	
	if( size <= 0 && (width <= 0 || height <= 0) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() requested size/dimensions are negative or zero");
		return NULL;
	}

	if( timestamp < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() timestamp cannot be negative");
		return NULL;
	}

	const bool isImage = (width > 0) && (height > 0);

	if( isImage && format == IMAGE_UNKNOWN )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() invalid format string");
		return NULL;
	}

	if( isImage )
		size = imageFormatSize(format, width, height);

	// allocate memory
	void* ptr = NULL;

    PYCUDA_ASSERT_NOGIL(cudaMallocMapped(&ptr, size, clear));
    
	return isImage ? PyCUDA_RegisterImage(ptr, width, height, format, timestamp, true)
                   : PyCUDA_RegisterMemory(ptr, size, true);
}


// PyCUDA_Memcpy (TODO variable size / offset)
PyObject* PyCUDA_Memcpy( PyObject* self, PyObject* args, PyObject* kwds )
{
	PyObject* dst_capsule = NULL;
	PyObject* src_capsule = NULL;
	
	int mapped = 1;
	cudaStream_t stream = 0;
	
	static char* kwlist[]  = {"dst", "src", "mapped", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|OpK", kwlist, &dst_capsule, &src_capsule, &mapped, &stream) )
		return NULL;

	// check if the args were reversed in the single-arg version
	if( !src_capsule && dst_capsule != NULL )
	{
		src_capsule = dst_capsule;
		dst_capsule = NULL;
	}
	
	// get the src image
	PyCudaMemory* src_mem = PyCUDA_GetMemory(src_capsule);
	PyCudaImage* src_img = PyCUDA_GetImage(src_capsule);
	
	if( !src_mem && !src_img )
	{
	    PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "src should either be a cudaImage or cudaMemory");
	    return NULL;
	}
	
	// allocate the dst image (if needed)
	bool dst_allocated = false;

	if( !dst_capsule )
	{
	    void* dst_ptr = NULL;

	    if( mapped )
	    {
	        PYCUDA_ASSERT_NOGIL(cudaMallocMapped(&dst_ptr, src_mem->size, false));
		}
		else
		{
		    PYCUDA_ASSERT_NOGIL(cudaMalloc(&dst_ptr, src_mem->size));
		}

		if( src_img != NULL )
		    dst_capsule = PyCUDA_RegisterImage(dst_ptr, src_img->width, src_img->height, src_img->format, src_img->timestamp, mapped);
		else
		    dst_capsule = PyCUDA_RegisterMemory(dst_ptr, src_mem->size, mapped);
		       
		dst_allocated = true;
	}
	
	PyCudaMemory* dst_mem = PyCUDA_GetMemory(dst_capsule);
	PyCudaImage* dst_img = PyCUDA_GetImage(dst_capsule);
	
	if( !dst_mem && !dst_img )
    {
        PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "dst should either be a cudaImage or cudaMemory");
        return NULL;
    }
    
    if( src_mem->size != dst_mem->size )
    {
        PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "src and dst need to have the same size");
        return NULL;
    }
    
    if( src_img && dst_img )
        dst_img->timestamp = src_img->timestamp;

	PYCUDA_ASSERT(cudaMemcpyAsync(dst_mem->ptr, src_mem->ptr, src_mem->size, cudaMemcpyDeviceToDevice, stream));

	if( dst_allocated )
		return dst_capsule;
	
	Py_RETURN_NONE;
}


// PyCUDA_StreamCreate
PyObject* PyCUDA_StreamCreate( PyObject* self, PyObject* args, PyObject* kwds )
{
	int nonblocking = 0;
	int priority = 0;
	
	static char* kwlist[]  = {"nonblocking", "priority", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|pi", kwlist, &nonblocking, &priority) )
		return NULL;
		
    cudaStream_t stream = 0;
    const uint32_t flags = nonblocking ? cudaStreamNonBlocking : cudaStreamDefault;

    PYCUDA_ASSERT_NOGIL(cudaStreamCreateWithPriority(&stream, flags, priority));

    return PYLONG_FROM_PTR(stream);
}


// PyCUDA_StreamDestroy
PyObject* PyCUDA_StreamDestroy( PyObject* self, PyObject* args )
{
	cudaStream_t stream = 0;

	if( !PyArg_ParseTuple(args, "K", &stream) )
		return NULL;
		
    PYCUDA_ASSERT_NOGIL(cudaStreamDestroy(stream));

    Py_RETURN_NONE;
}

	
// PyCUDA_StreamSynchronize
PyObject* PyCUDA_StreamSynchronize( PyObject* self, PyObject* args )
{
	cudaStream_t stream = 0;

	if( !PyArg_ParseTuple(args, "K", &stream) )
		return NULL;
		
    PYCUDA_ASSERT_NOGIL(cudaStreamSynchronize(stream));
    
    Py_RETURN_NONE;
}


// PyCUDA_StreamWaitEvent
PyObject* PyCUDA_StreamWaitEvent( PyObject* self, PyObject* args )
{
	cudaStream_t stream = 0;
	cudaEvent_t event = 0;

	if( !PyArg_ParseTuple(args, "KK", &stream, &event) )
		return NULL;
		
    PYCUDA_ASSERT(cudaStreamWaitEvent(stream, event, 0));
    
    Py_RETURN_NONE;
}


// PyCUDA_EventCreate
PyObject* PyCUDA_EventCreate( PyObject* self )
{
    cudaEvent_t event = 0;
	PYCUDA_ASSERT(cudaEventCreate(&event));
	return PYLONG_FROM_PTR(event);
}


// PyCUDA_EventDestroy
PyObject* PyCUDA_EventDestroy( PyObject* self, PyObject* args )
{
	cudaEvent_t event = 0;

	if( !PyArg_ParseTuple(args, "K", &event) )
		return NULL;
		
    PYCUDA_ASSERT(cudaEventDestroy(event));

    Py_RETURN_NONE;
}

	
// PyCUDA_EventRecord
PyObject* PyCUDA_EventRecord( PyObject* self, PyObject* args, PyObject* kwds )
{
	cudaEvent_t event = 0;
    cudaStream_t stream = 0;
    
	static char* kwlist[] = {"event", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|KK", kwlist, &event, &stream) )
		return NULL;
		
    if( !event )
        PYCUDA_ASSERT(cudaEventCreate(&event));
        
    PYCUDA_ASSERT(cudaEventRecord(event, stream));

    return PYLONG_FROM_PTR(event);
}


// PyCUDA_EventQuery
PyObject* PyCUDA_EventQuery( PyObject* self, PyObject* args )
{
	cudaEvent_t event = 0;

	if( !PyArg_ParseTuple(args, "K", &event) )
		return NULL;
		
    const cudaError_t retval = cudaEventQuery(event);
    
    if( retval == cudaSuccess )
    {
        Py_RETURN_TRUE;
    }
    else if( retval == cudaErrorNotReady )
    {
        Py_RETURN_FALSE;
    }
    else
    {
        PyErr_Format(PyExc_Exception, "CUDA error (%u) - %s\n  File \"%s\", line %i\n    cudaEventQuery(event);", retval, cudaGetErrorString(retval), __FILE__, __LINE__);
        return NULL;
    }
}


// PyCUDA_EventElapsedTime
PyObject* PyCUDA_EventElapsedTime( PyObject* self, PyObject* args )
{
    cudaEvent_t start, end = 0;
    float ms = 0.0f;
    
	if( !PyArg_ParseTuple(args, "KK", &start, &end) )
		return NULL;
		
    const cudaError_t retval = cudaEventElapsedTime(&ms, start, end);
    
    if( retval == cudaSuccess )
    {
        return PyFloat_FromDouble(ms);
    }
    else if( retval == cudaErrorNotReady )
    {
        return PyFloat_FromDouble(-1.0);
    }
    else
    {
        PyErr_Format(PyExc_Exception, "CUDA error (%u) - %s\n  File \"%s\", line %i\n    cudaEventElapsedTime(&ms, start, end)", retval, cudaGetErrorString(retval), __FILE__, __LINE__);
        return NULL;
    }	
}


// PyCUDA_EventSynchronize
PyObject* PyCUDA_EventSynchronize( PyObject* self, PyObject* args )
{
	cudaEvent_t event = 0;

	if( !PyArg_ParseTuple(args, "K", &event) )
		return NULL;
		
    PYCUDA_ASSERT_NOGIL(cudaEventSynchronize(event));

    Py_RETURN_NONE;
}

	
// PyCUDA_DeviceSynchronize
PyObject* PyCUDA_DeviceSynchronize( PyObject* self )
{
	PYCUDA_ASSERT_NOGIL(cudaDeviceSynchronize());
	Py_RETURN_NONE;
}


// PyCUDA_AdaptFontSize
PyObject* PyCUDA_AdaptFontSize( PyObject* self, PyObject* args )
{
	int dim = 0;

	if( !PyArg_ParseTuple(args, "i", &dim) )
		return NULL;

	if( dim <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "adaptFontSize() requested size is negative or zero");
		return NULL;
	}

	return PyFloat_FromDouble(adaptFontSize(dim));
}


// PyCUDA_ConvertColor
PyObject* PyCUDA_ConvertColor( PyObject* self, PyObject* args, PyObject* kwds )
{
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;
	cudaStream_t stream = 0;
	
	static char* kwlist[] = {"input", "output", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO|K", kwlist, &pyInput, &pyOutput, &stream))
		return NULL;
	
	// get pointers to image data
	PyCudaImage* input = PyCUDA_GetImage(pyInput);
	PyCudaImage* output = PyCUDA_GetImage(pyOutput);

	if( !input || !output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaConvertColor() failed to get input/output image pointers (should be cudaImage)");
		return NULL;
	}

	if( input->width != output->width || input->height != output->height )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaConvertColor() input and output image resolutions are different");
		return NULL;
	}

	// run the CUDA function
	PYCUDA_ASSERT_NOGIL(cudaConvertColor(input->base.ptr, input->format, output->base.ptr, output->format, input->width, input->height, stream));

	output->timestamp = input->timestamp;

	// return void
	Py_RETURN_NONE;
}


// PyCUDA_Resize
PyObject* PyCUDA_Resize( PyObject* self, PyObject* args, PyObject* kwds )
{
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;

	const char* filter_str = "point";
	cudaStream_t stream = 0;
	
	static char* kwlist[] = {"input", "output", "filter", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO|sK", kwlist, &pyInput, &pyOutput, &filter_str, &stream))
		return NULL;

	const cudaFilterMode filter_mode = cudaFilterModeFromStr(filter_str);
	
	// get pointers to image data
	PyCudaImage* input = PyCUDA_GetImage(pyInput);
	PyCudaImage* output = PyCUDA_GetImage(pyOutput);

	if( !input || !output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaResize() failed to get input/output image pointers (should be cudaImage)");
		return NULL;
	}

	if( input->format != output->format )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaResize() input and output image formats are different");
		return NULL;
	}

	// run the CUDA function
	PYCUDA_ASSERT_NOGIL(cudaResize(input->base.ptr, input->width, input->height, output->base.ptr, output->width, output->height, output->format, filter_mode, stream));

	output->timestamp = input->timestamp;

	// return void
	Py_RETURN_NONE;
}


// PyCUDA_Crop
PyObject* PyCUDA_Crop( PyObject* self, PyObject* args, PyObject* kwds )
{
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;
	
	float left, top, right, bottom;
	cudaStream_t stream = 0;
	
	static char* kwlist[] = {"input", "output", "roi", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO(ffff)|K", kwlist, &pyInput, &pyOutput, &left, &top, &right, &bottom, &stream))
		return NULL;

	// get pointers to image data
	PyCudaImage* input = PyCUDA_GetImage(pyInput);
	PyCudaImage* output = PyCUDA_GetImage(pyOutput);

	if( !input || !output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaCrop() failed to get input/output image pointers (should be cudaImage)");
		return NULL;
	}

	if( input->format != output->format )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaCrop() input and output image formats are different");
		return NULL;
	}

	// validate ROI
	const float roi_width  = right - left;
	const float roi_height = bottom - top;

	if( left < 0 || top < 0 || right < 0 || bottom < 0 ||
	    right > input->width || bottom > input->height ||
	    roi_width <= 0 || roi_height <= 0 || 
	    roi_width > output->width || roi_height > output->height )
	{
		PyErr_SetString(PyExc_ValueError, LOG_PY_UTILS "cudaCrop() had an invalid ROI");
		return NULL;
	}

	// run the CUDA function
	PYCUDA_ASSERT_NOGIL(cudaCrop(input->base.ptr, output->base.ptr, make_int4(left, top, right, bottom), input->width, input->height, input->format, stream));

	output->timestamp = input->timestamp;

	// return void
	Py_RETURN_NONE;
}


// PyCUDA_Normalize
PyObject* PyCUDA_Normalize( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;
	
	float input_min, input_max;
	float output_min, output_max;
	
	cudaStream_t stream = NULL;
	
	static char* kwlist[] = {"input", "inputRange", "output", "outputRange", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O(ff)O(ff)|K", kwlist, &pyInput, &input_min, &input_max, &pyOutput, &output_min, &output_max, &stream))
		return NULL;

	// get pointers to image data
	PyCudaImage* input = PyCUDA_GetImage(pyInput);
	PyCudaImage* output = PyCUDA_GetImage(pyOutput);

	if( !input || !output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaNormalize() failed to get input/output image pointers (should be cudaImage)");
		return NULL;
	}

	if( input->format != output->format )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaNormalize() input and output image formats are different");
		return NULL;
	}

	if( input->width != output->width || input->height != output->height )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaNormalize() input and output image resolutions are different");
		return NULL;
	}

	// run the CUDA function
	PYCUDA_ASSERT_NOGIL(cudaNormalize(input->base.ptr, make_float2(input_min, input_max), output->base.ptr, make_float2(output_min, output_max), output->width, output->height, output->format, stream));

	output->timestamp = input->timestamp;

	// return void
	Py_RETURN_NONE;
}


// PyCUDA_Overlay
PyObject* PyCUDA_Overlay( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;

	float x = 0.0f;
	float y = 0.0f;

    cudaStream_t stream = 0;
    
	static char* kwlist[] = {"input", "output", "x", "y", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO|ffK", kwlist, &pyInput, &pyOutput, &x, &y, &stream))
		return NULL;

	// get pointers to image data
	PyCudaImage* input = PyCUDA_GetImage(pyInput);
	PyCudaImage* output = PyCUDA_GetImage(pyOutput);

	if( !input || !output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaOverlay() failed to get input/output image pointers (should be cudaImage)");
		return NULL;
	}

	if( input->format != output->format )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaOverlay() input and output image formats are different");
		return NULL;
	}

	// run the CUDA function
	PYCUDA_ASSERT_NOGIL(cudaOverlay(input->base.ptr, input->width, input->height, output->base.ptr, output->width, output->height, output->format, x, y, stream));

	output->timestamp = input->timestamp;

	// return void
	Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------
// PyCUDA_DrawCircle
PyObject* PyCUDA_DrawCircle( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;
	PyObject* pyColor  = NULL;
	
	float x = 0.0f;
	float y = 0.0f;
	float radius = 0.0f;
	
	cudaStream_t stream = 0;
	
	static char* kwlist[] = {"input", "center", "radius", "color", "output", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O(ff)fO|OK", kwlist, &pyInput, &x, &y, &radius, &pyColor, &pyOutput, &stream))
		return NULL;
	
	if( !pyOutput )
		pyOutput = pyInput;
	
	// get pointers to image data
	PyCudaImage* input = PyCUDA_GetImage(pyInput);
	PyCudaImage* output = PyCUDA_GetImage(pyOutput);

	if( !input || !output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to get input/output CUDA image pointers (should be cudaImage)");
		return NULL;
	}

	if( input->width != output->width || input->height != output->height || input->format != output->format )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "input/output images need to have matching dimensions and formats");
		return NULL;
	}	
	
	// parse the color
	float4 color = make_float4(0, 0, 0, 255);
	
	if( !PyTuple_Check(pyColor) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "color argument isn't a valid tuple");
		return NULL;
	}

	if( !PyArg_ParseTuple(pyColor, "fff|f", &color.x, &color.y, &color.z, &color.w) )
		return NULL;

	// run the CUDA function
	PYCUDA_ASSERT_NOGIL(cudaDrawCircle(input->base.ptr, output->base.ptr, input->width, input->height,
							           input->format, x, y, radius, color, stream));

	output->timestamp = input->timestamp;

	Py_RETURN_NONE;
}


// PyCUDA_DrawLine
PyObject* PyCUDA_DrawLine( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;
	PyObject* pyColor  = NULL;
	
	float x1 = 0.0f;
	float y1 = 0.0f;
	float x2 = 0.0f;
	float y2 = 0.0f;
	
	float line_width = 1.0f;
    cudaStream_t stream = 0;
    
	static char* kwlist[] = {"input", "a", "b", "color", "line_width", "output", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O(ff)(ff)O|fOK", kwlist, &pyInput, &x1, &y1, &x2, &y2, &pyColor, &line_width, &pyOutput, &stream))
		return NULL;

	if( !pyOutput )
		pyOutput = pyInput;
	
	// get pointers to image data
	PyCudaImage* input = PyCUDA_GetImage(pyInput);
	PyCudaImage* output = PyCUDA_GetImage(pyOutput);

	if( !input || !output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to get input/output CUDA image pointers (should be cudaImage)");
		return NULL;
	}

	if( input->width != output->width || input->height != output->height || input->format != output->format )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "input/output images need to have matching dimensions and formats");
		return NULL;
	}	
	
	// parse the color
	float4 color = make_float4(0, 0, 0, 255);
	
	if( !PyTuple_Check(pyColor) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "color argument isn't a valid tuple");
		return NULL;
	}

	if( !PyArg_ParseTuple(pyColor, "fff|f", &color.x, &color.y, &color.z, &color.w) )
		return NULL;

	// run the CUDA function
	PYCUDA_ASSERT_NOGIL(cudaDrawLine(input->base.ptr, output->base.ptr, input->width, input->height,
						             input->format, x1, y1, x2, y2, color, line_width, stream));

	output->timestamp = input->timestamp;

	Py_RETURN_NONE;
}


// PyCUDA_DrawRect
PyObject* PyCUDA_DrawRect( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;
	PyObject* pyColor  = NULL;
	PyObject* pyLineColor = NULL;
	
	float left = 0.0f;
	float top = 0.0f;
	float right = 0.0f;
	float bottom = 0.0f;
	
	float line_width = 1.0f;
	cudaStream_t stream = 0;
	
	static char* kwlist[] = {"input", "rect", "color", "line_color", "line_width", "output", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O(ffff)|OOfOK", kwlist, &pyInput, &left, &top, &right, &bottom, &pyColor, &pyLineColor, &line_width, &pyOutput, &stream))
		return NULL;

	if( !pyOutput )
		pyOutput = pyInput;
	
	// get pointers to image data
	PyCudaImage* input = PyCUDA_GetImage(pyInput);
	PyCudaImage* output = PyCUDA_GetImage(pyOutput);

	if( !input || !output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to get input/output CUDA image pointers (should be cudaImage)");
		return NULL;
	}

	if( input->width != output->width || input->height != output->height || input->format != output->format )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "input/output images need to have matching dimensions and formats");
		return NULL;
	}	
	
	// parse the color
	float4 color = make_float4(0,0,0,0);

	if( pyColor != NULL && PyTuple_Check(pyColor) )
	{
		if( !PyArg_ParseTuple(pyColor, "fff|f", &color.x, &color.y, &color.z, &color.w) )
			return NULL;
	}
	else if( pyColor != NULL )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "color argument isn't a valid tuple");
		return NULL;
	}

	// parse the line color
	float4 line_color = make_float4(0,0,0,0);
	
	if( pyLineColor != NULL && PyTuple_Check(pyLineColor) )
	{
		if( !PyArg_ParseTuple(pyLineColor, "fff|f", &line_color.x, &line_color.y, &line_color.z, &line_color.w) )
			return NULL;
	}
	else if( pyLineColor != NULL )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "line_color argument isn't a valid tuple");
		return NULL;
	}

	// run the CUDA function
	PYCUDA_ASSERT_NOGIL(cudaDrawRect(input->base.ptr, output->base.ptr, input->width, input->height, input->format,
						             left, top, right, bottom, color, line_color, line_width, stream));
						             
	output->timestamp = input->timestamp;

	Py_RETURN_NONE;
}


//-------------------------------------------------------------------------------
// PyFont container
typedef struct {
	PyObject_HEAD
	cudaFont* font;

	// colors
	PyObject* black;
	PyObject* white;
	PyObject* gray;
	PyObject* brown;
	PyObject* tan;
	PyObject* red;
	PyObject* green;
	PyObject* blue;
	PyObject* cyan;
	PyObject* lime;
	PyObject* yellow;
	PyObject* orange;
	PyObject* purple;
	PyObject* magenta;

	PyObject* gray_90;
	PyObject* gray_80;
	PyObject* gray_70;
	PyObject* gray_60;
	PyObject* gray_50;
	PyObject* gray_40;
	PyObject* gray_30;
	PyObject* gray_20;
	PyObject* gray_10;

} PyFont_Object;


// New
static PyObject* PyFont_New( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyFont_New()\n");
	
	// allocate a new container
	PyFont_Object* self = (PyFont_Object*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_UTILS "cudaFont tp_alloc() failed to allocate a new object");
		LogDebug(LOG_PY_UTILS "cudaFont tp_alloc() failed to allocate a new object\n");
		return NULL;
	}
	

	#define INIT_COLOR(color, r, g, b)		\
		self->color = Py_BuildValue("(ffff)", r, g, b, 255.0);	\
		if( !self->color ) {							\
			Py_DECREF(self);							\
			return NULL;								\
		}

	#define INIT_GRAY(color, a)		\
		self->color = Py_BuildValue("(ffff)", 0.0, 0.0, 0.0, a);	\
		if( !self->color ) {							\
			Py_DECREF(self);							\
			return NULL;								\
		}

	INIT_COLOR(black,   0.0, 0.0, 0.0);									
	INIT_COLOR(white,   255.0, 255.0, 255.0);
	INIT_COLOR(gray,	128.0, 128.0, 128.0);
	INIT_COLOR(brown,	165.0, 42.0, 42.0);
	INIT_COLOR(tan,	210.0, 180.0, 140.0);
	INIT_COLOR(red,     255.0, 255.0, 255.0);
	INIT_COLOR(green,   0.0, 200.0, 128.0);
	INIT_COLOR(blue,    0.0, 0.0, 255.0);
	INIT_COLOR(cyan,    0.0, 255.0, 255.0);
	INIT_COLOR(lime,    0.0, 255.0, 0.0);
	INIT_COLOR(yellow,  255.0, 255.0, 0.0);
	INIT_COLOR(orange,  255.0, 165.0, 0.0);
	INIT_COLOR(purple,  128.0, 0.0, 128.0);
	INIT_COLOR(magenta, 255.0, 0.0, 255.0);

	INIT_GRAY(gray_90, 230.0);
	INIT_GRAY(gray_80, 200.0);
	INIT_GRAY(gray_70, 180.0);
	INIT_GRAY(gray_60, 150.0);
	INIT_GRAY(gray_50, 127.5);
	INIT_GRAY(gray_40, 100.0);
	INIT_GRAY(gray_30, 75.0);
	INIT_GRAY(gray_20, 50.0);
	INIT_GRAY(gray_10, 25.0);

	self->font = NULL;
	return (PyObject*)self;
}


// Init
static int PyFont_Init( PyFont_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyFont_Init()\n");
	
	// parse arguments
	const char* font_name = NULL;
	float font_size = 32.0f;

	static char* kwlist[] = {"font", "size", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sf", kwlist, &font_name, &font_size))
		return -1;

	// create the font
	cudaFont* font = cudaFont::Create(font_name, font_size);

	if( !font )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to create cudaFont object");
		return -1;
	}

	self->font = font;
	return 0;
}


// Deallocate
static void PyFont_Dealloc( PyFont_Object* self )
{
	LogDebug(LOG_PY_UTILS "PyFont_Dealloc()\n");

	// free the font
	if( self->font != NULL )
	{
		delete self->font;
		self->font = NULL;
	}
	
	// free the color objects
	Py_XDECREF(self->black);
	Py_XDECREF(self->white);
	Py_XDECREF(self->gray);
	Py_XDECREF(self->brown);
	Py_XDECREF(self->tan);
	Py_XDECREF(self->red);
	Py_XDECREF(self->green);
	Py_XDECREF(self->blue);
	Py_XDECREF(self->cyan);
	Py_XDECREF(self->lime);
	Py_XDECREF(self->yellow);
	Py_XDECREF(self->orange);
	Py_XDECREF(self->purple);
	Py_XDECREF(self->magenta);

	Py_XDECREF(self->gray_90);
	Py_XDECREF(self->gray_80);
	Py_XDECREF(self->gray_70);
	Py_XDECREF(self->gray_60);
	Py_XDECREF(self->gray_50);
	Py_XDECREF(self->gray_40);
	Py_XDECREF(self->gray_30);
	Py_XDECREF(self->gray_20);
	Py_XDECREF(self->gray_10);

	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}


// Overlay
static PyObject* PyFont_OverlayText( PyFont_Object* self, PyObject* args, PyObject* kwds )
{
	if( !self || !self->font )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont invalid object instance");
		return NULL;
	}

	// parse arguments
	PyObject* input  = NULL;
	PyObject* color  = NULL;
	PyObject* bg     = NULL;

	const char* text = NULL;
	const char* format_str = "rgba32f";

	int width = 0;
	int height = 0;
    int padding = 5;
    
	int x = 0;
	int y = 0;

    cudaStream_t stream = 0;
    
	static char* kwlist[] = {"image", "width", "height", "text", "x", "y", "color", "background", "format", "padding", "stream", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|iisiiOOsiK", kwlist, &input, &width, &height, &text, &x, &y, &color, &bg, &format_str, &padding, &stream))
		return NULL;

	// make sure that text exists
	if( !text )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() needs to be called with the 'text' string argument");
		return NULL;
	}

	// parse color tuple
	float4 rgba = make_float4(0, 0, 0, 255);

	if( color != NULL )
	{
		if( !PyTuple_Check(color) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() color argument isn't a valid tuple");
			return NULL;
		}

		if( !PyArg_ParseTuple(color, "fff|f", &rgba.x, &rgba.y, &rgba.z, &rgba.w) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() failed to parse color tuple");
			return NULL;
		}
	}

	// parse background color tuple
	float4 bg_rgba = make_float4(0, 0, 0, 0);

	if( bg != NULL )
	{
		if( !PyTuple_Check(bg) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() background color argument isn't a valid tuple");
			return NULL;
		}

		if( !PyArg_ParseTuple(bg, "fff|f", &bg_rgba.x, &bg_rgba.y, &bg_rgba.z, &bg_rgba.w) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() failed to parse background color tuple");
			return NULL;
		}
	}

	// parse format string
	imageFormat format = imageFormatFromStr(format_str);

	// get pointer to image data
	void* ptr = PyCUDA_GetImage(input, &width, &height, &format);

	if( !ptr )
		return NULL;

	// render the font overlay
	bool result = false;
	Py_BEGIN_ALLOW_THREADS
	result = self->font->OverlayText(ptr, format, width, height, text, x, y, rgba, bg_rgba, padding, stream);
    Py_END_ALLOW_THREADS
    
    if( !result )
    {
        PyErr_Format(PyExc_Exception, LOG_PY_UTILS "cudaFont failed to render text \"%s\"", text);
        return NULL;
    }
    
	// return void
	Py_RETURN_NONE;
}

// GetSize
static PyObject* PyFont_GetSize( PyFont_Object* self )
{
	if( !self || !self->font )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont invalid object instance");
		return NULL;
	}
	
	//return PyFloat_FromDouble(self->font->GetSize());
	return PYLONG_FROM_UNSIGNED_LONG(self->font->GetSize());
}

// TextExtents
static PyObject* PyFont_TextExtents( PyFont_Object* self, PyObject* args, PyObject* kwds )
{
	if( !self || !self->font )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont invalid object instance");
		return NULL;
	}
	
    // parse arguments
	const char* text = NULL;

	int x = 0;
	int y = 0;

	static char* kwlist[] = {"text", "x", "y", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "s|ii", kwlist, &text) )
		return NULL;

	// make sure that text exists
	if( !text )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.TextExtents() needs to be called with the 'text' string argument");
		return NULL;
	}
	
	const int4 extents = self->font->TextExtents(text, x, y);
	
    PyObject* x1 = PYLONG_FROM_LONG(extents.x);
    PyObject* y1 = PYLONG_FROM_LONG(extents.y);
    PyObject* x2 = PYLONG_FROM_LONG(extents.z);
    PyObject* y2 = PYLONG_FROM_LONG(extents.w);

	PyObject* tuple = PyTuple_Pack(4, x1, y1, x2, y2);

	Py_DECREF(x1);
	Py_DECREF(y1);
	Py_DECREF(x2);
	Py_DECREF(y2);

	return tuple;
}

static PyTypeObject pyFont_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyFont_Methods[] = 
{
	{ "OverlayText", (PyCFunction)PyFont_OverlayText, METH_VARARGS|METH_KEYWORDS, "Render the font overlay for a given text string"},
	{ "TextExtents", (PyCFunction)PyFont_TextExtents, METH_VARARGS|METH_KEYWORDS, "Return the bounding rectangle of the given text string"},
	{ "GetSize", (PyCFunction)PyFont_GetSize, METH_NOARGS, "Return the size of the font (height in pixels)"},
	{NULL}  /* Sentinel */
};

static PyMemberDef pyFont_Members[] = 
{
	{ "Black",   T_OBJECT_EX, offsetof(PyFont_Object, black),   0, "Black color tuple"},
	{ "White",   T_OBJECT_EX, offsetof(PyFont_Object, white),   0, "White color tuple"},
	{ "Gray",    T_OBJECT_EX, offsetof(PyFont_Object, gray),    0, "Gray color tuple"},
	{ "Brown",   T_OBJECT_EX, offsetof(PyFont_Object, brown),   0, "Brown color tuple"},
	{ "Tan",     T_OBJECT_EX, offsetof(PyFont_Object, tan),     0, "Tan color tuple"},
	{ "Red",     T_OBJECT_EX, offsetof(PyFont_Object, red),     0, "Red color tuple"},
	{ "Green",   T_OBJECT_EX, offsetof(PyFont_Object, green),   0, "Green color tuple"},
	{ "Blue",    T_OBJECT_EX, offsetof(PyFont_Object, blue),    0, "Blue color tuple"},
	{ "Cyan",    T_OBJECT_EX, offsetof(PyFont_Object, cyan),    0, "Cyan color tuple"},
	{ "Lime",    T_OBJECT_EX, offsetof(PyFont_Object, lime),    0, "Lime color tuple"},
	{ "Yellow",  T_OBJECT_EX, offsetof(PyFont_Object, yellow),  0, "Yellow color tuple"},
	{ "Orange",  T_OBJECT_EX, offsetof(PyFont_Object, orange),  0, "Orange color tuple"},
	{ "Purple",  T_OBJECT_EX, offsetof(PyFont_Object, purple),  0, "Purple color tuple"},
	{ "Magenta", T_OBJECT_EX, offsetof(PyFont_Object, magenta), 0, "Magenta color tuple"},
	{ "Gray90",  T_OBJECT_EX, offsetof(PyFont_Object, gray_90), 0, "Gray color tuple (90% alpha)"},
	{ "Gray80",  T_OBJECT_EX, offsetof(PyFont_Object, gray_80), 0, "Gray color tuple (80% alpha)"},
	{ "Gray70",  T_OBJECT_EX, offsetof(PyFont_Object, gray_70), 0, "Gray color tuple (70% alpha)"},
	{ "Gray60",  T_OBJECT_EX, offsetof(PyFont_Object, gray_60), 0, "Gray color tuple (60% alpha)"},
	{ "Gray50",  T_OBJECT_EX, offsetof(PyFont_Object, gray_50), 0, "Gray color tuple (50% alpha)"},
	{ "Gray40",  T_OBJECT_EX, offsetof(PyFont_Object, gray_40), 0, "Gray color tuple (40% alpha)"},
	{ "Gray30",  T_OBJECT_EX, offsetof(PyFont_Object, gray_30), 0, "Gray color tuple (30% alpha)"},
	{ "Gray20",  T_OBJECT_EX, offsetof(PyFont_Object, gray_20), 0, "Gray color tuple (20% alpha)"},
	{ "Gray10",  T_OBJECT_EX, offsetof(PyFont_Object, gray_10), 0, "Gray color tuple (10% alpha)"},
	{NULL}  /* Sentinel */
};

bool PyFont_RegisterType( PyObject* module )
{
	if( !module )
		return false;

	// register font
	pyFont_Type.tp_name 	= PY_UTILS_MODULE_NAME ".cudaFont";
	pyFont_Type.tp_basicsize = sizeof(PyFont_Object);
	pyFont_Type.tp_flags 	= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyFont_Type.tp_methods   = pyFont_Methods;
	pyFont_Type.tp_members	= pyFont_Members;
	pyFont_Type.tp_new 	     = PyFont_New;
	pyFont_Type.tp_init	     = (initproc)PyFont_Init;
	pyFont_Type.tp_dealloc	= (destructor)PyFont_Dealloc;
	pyFont_Type.tp_doc  	= "Bitmap font overlay rendering with CUDA";
	 
	if( PyType_Ready(&pyFont_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "cudaFont PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyFont_Type);
    
	if( PyModule_AddObject(module, "cudaFont", (PyObject*)&pyFont_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "cudaFont PyModule_AddObject('cudaFont') failed\n");
		return false;
	}

	return true;
}

// PyLog_Usage
PyObject* PyLog_Usage( PyObject* self )
{
	return Py_BuildValue("s", Log::Usage());
}

//-------------------------------------------------------------------------------
static PyMethodDef pyCUDA_Functions[] = 
{
	{ "cudaMalloc", (PyCFunction)PyCUDA_Malloc, METH_VARARGS|METH_KEYWORDS, "Allocated CUDA memory on the GPU with cudaMalloc()" },
	{ "cudaMallocMapped", (PyCFunction)PyCUDA_AllocMapped, METH_VARARGS|METH_KEYWORDS, "Allocate CUDA ZeroCopy mapped memory" },
	{ "cudaAllocMapped", (PyCFunction)PyCUDA_AllocMapped, METH_VARARGS|METH_KEYWORDS, "Allocate CUDA ZeroCopy mapped memory" },
	{ "cudaMemcpy", (PyCFunction)PyCUDA_Memcpy, METH_VARARGS|METH_KEYWORDS, "Copy src image to dst image (or if dst is not provided, return a new image with the contents of src)" },
	{ "cudaStreamCreate", (PyCFunction)PyCUDA_StreamCreate, METH_VARARGS|METH_KEYWORDS, "Create a new CUDA stream, by default non-blocking and with priority 0 (otherwise set kwargs blocking=True priority=int)" },
	{ "cudaStreamDestroy", (PyCFunction)PyCUDA_StreamDestroy, METH_VARARGS, "Destroy the given CUDA stream" },
	{ "cudaStreamWaitEvent", (PyCFunction)PyCUDA_StreamWaitEvent, METH_VARARGS, "Make the stream wait for the event before continuing" },
	{ "cudaStreamSynchronize", (PyCFunction)PyCUDA_StreamSynchronize, METH_VARARGS, "Synchronize with the given CUDA stream" },
	{ "cudaEventCreate", (PyCFunction)PyCUDA_EventCreate, METH_NOARGS, "Create a new CUDA event" },
	{ "cudaEventDestroy", (PyCFunction)PyCUDA_EventDestroy, METH_VARARGS, "Destroy the given CUDA event" },
	{ "cudaEventRecord", (PyCFunction)PyCUDA_EventRecord, METH_VARARGS|METH_KEYWORDS, "Record the event to the CUDA stream (if the event isn't provided, it will be created first and then returned)" },
	{ "cudaEventQuery", (PyCFunction)PyCUDA_EventQuery, METH_VARARGS, "Return True if the event has completed execution, otherwise False" },
	{ "cudaEventElapsedTime", (PyCFunction)PyCUDA_EventElapsedTime, METH_VARARGS, "Return the elapsed time between two events in milliseconds, or -1 if either event hasn't completed execution yet." },
	{ "cudaEventSynchronize", (PyCFunction)PyCUDA_EventSynchronize, METH_VARARGS, "Wait on the CPU for the event to be finished" },
	{ "cudaDeviceSynchronize", (PyCFunction)PyCUDA_DeviceSynchronize, METH_NOARGS, "Wait for the GPU to complete all work" },
	{ "cudaConvertColor", (PyCFunction)PyCUDA_ConvertColor, METH_VARARGS|METH_KEYWORDS, "Perform colorspace conversion on the GPU" },
	{ "cudaCrop", (PyCFunction)PyCUDA_Crop, METH_VARARGS|METH_KEYWORDS, "Crop an image on the GPU" },		
	{ "cudaResize", (PyCFunction)PyCUDA_Resize, METH_VARARGS|METH_KEYWORDS, "Resize an image on the GPU" },
	{ "cudaNormalize", (PyCFunction)PyCUDA_Normalize, METH_VARARGS|METH_KEYWORDS, "Normalize the pixel intensities of an image between two ranges" },
	{ "cudaOverlay", (PyCFunction)PyCUDA_Overlay, METH_VARARGS|METH_KEYWORDS, "Overlay the input image onto the composite output image at position (x,y)" },
	{ "cudaDrawCircle", (PyCFunction)PyCUDA_DrawCircle, METH_VARARGS|METH_KEYWORDS, "Draw a circle with the specified radius and color centered at position (x,y)" },
	{ "cudaDrawLine", (PyCFunction)PyCUDA_DrawLine, METH_VARARGS|METH_KEYWORDS, "Draw a line with the specified color and line width from (x1,y1) to (x2,y2)" },
	{ "cudaDrawRect", (PyCFunction)PyCUDA_DrawRect, METH_VARARGS|METH_KEYWORDS, "Draw a rect with the specified color at (left, top, right, bottom)" },
	{ "adaptFontSize", (PyCFunction)PyCUDA_AdaptFontSize, METH_VARARGS, "Determine an appropriate font size for the given image dimension" },
	{ "logUsage", (PyCFunction)PyLog_Usage, METH_NOARGS, "Return help text describing the command line arguments of the logging interface" },
	{NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyCUDA_RegisterFunctions()
{
	return pyCUDA_Functions;
}

// Register CUDA types
bool PyCUDA_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;
	
	if( !PyCudaMemory_RegisterType(module) )
		return false;

	if( !PyCudaImage_RegisterType(module) )
		return false;

	if( !PyFont_RegisterType(module) )
		return false;

	return true;
}

