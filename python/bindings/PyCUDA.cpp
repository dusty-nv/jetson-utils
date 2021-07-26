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
		LogError(LOG_PY_UTILS "cudaMemory tp_alloc() failed to allocate a new object\n");
		return NULL;
	}
	
	self->ptr = NULL;
	self->size = 0;
	self->mapped = false;
	self->freeOnDelete = true;

	return (PyObject*)self;
}

// PyCudaMemory_Dealloc
static void PyCudaMemory_Dealloc( PyCudaMemory* self )
{
	LogDebug(LOG_PY_UTILS "PyCudaMemory_Dealloc()\n");
	
	if( self->freeOnDelete && self->ptr != NULL )
	{
		if( self->mapped )
			CUDA(cudaFreeHost(self->ptr));
		else
			CUDA(cudaFree(self->ptr));

		self->ptr = NULL;
	}

	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}

// PyCudaMemory_Init
static int PyCudaMemory_Init( PyCudaMemory* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyCudaMemory_Init()\n");
	
	// parse arguments
	int size = 0;
	int mapped = 1;
	int freeOnDelete = 1;

	static char* kwlist[] = {"size", "mapped", "freeOnDelete", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "i|ii", kwlist, &size, &mapped, &freeOnDelete))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMemory.__init()__ failed to parse args tuple");
		LogDebug(LOG_PY_UTILS "cudaMemory.__init()__ failed to parse args tuple\n");
		return -1;
	}
    
	if( size < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMemory.__init()__ had invalid size");
		return -1;
	}

	// allocate CUDA memory
	if( mapped > 0 )
	{
		if( !cudaAllocMapped(&self->ptr, size) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMemory.__init()__ failed to allocate CUDA mapped memory");
			return -1;
		}
	}
	else
	{
		if( CUDA_FAILED(cudaMalloc(&self->ptr, size)) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMemory.__init()__ failed to allocate CUDA memory");
			return -1;
		}
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

	sprintf(str, 
		   "<cudaMemory object>\n"
		   "   -- ptr:    %p\n"
		   "   -- size:   %zu\n"
		   "   -- mapped: %s\n"
		   "   -- freeOnDelete: %s\n",
		   self->ptr, self->size, 
		   self->mapped ? "true" : "false", 
		   self->freeOnDelete ? "true" : "false");

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

static PyGetSetDef pyCudaMemory_GetSet[] = 
{
	{ "ptr", (getter)PyCudaMemory_GetPtr, NULL, "Address of CUDA memory", NULL},
	{ "size", (getter)PyCudaMemory_GetSize, NULL, "Size (in bytes)", NULL},
	{ "mapped", (getter)PyCudaMemory_GetMapped, NULL, "Is the memory mapped to CPU also? (zeroCopy)", NULL},
	{ "freeOnDelete", (getter)PyCudaMemory_GetFreeOnDelete, NULL, "Will the CUDA memory be released when the Python object is deleted?", NULL},	
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
	pyCudaMemory_Type.tp_methods    = NULL;
	pyCudaMemory_Type.tp_getset     = pyCudaMemory_GetSet;
	pyCudaMemory_Type.tp_new 	  = PyCudaMemory_New;
	pyCudaMemory_Type.tp_init	  = (initproc)PyCudaMemory_Init;
	pyCudaMemory_Type.tp_dealloc	  = (destructor)PyCudaMemory_Dealloc;
	pyCudaMemory_Type.tp_str		  = (reprfunc)PyCudaMemory_ToString;
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
		LogError(LOG_PY_UTILS "cudaImage tp_alloc() failed to allocate a new object\n");
		return NULL;
	}
	
	self->base.ptr = NULL;
	self->base.size = 0;
	self->base.mapped = false;
	self->base.freeOnDelete = true;
	
	self->width = 0;
	self->height = 0;
	
	self->shape[0] = 0; self->shape[1] = 0; self->shape[2] = 0;
	self->strides[0] = 0; self->strides[1] = 0; self->strides[2] = 0;
	
	self->format = IMAGE_UNKNOWN;
	
	return (PyObject*)self;
}

// PyCudaImage_Config
static void PyCudaImage_Config( PyCudaImage* self, void* ptr, uint32_t width, uint32_t height, imageFormat format, bool mapped, bool freeOnDelete )
{
	self->base.ptr = ptr;
	self->base.size = imageFormatSize(format, width, height);
	self->base.mapped = mapped;
	self->base.freeOnDelete = freeOnDelete;

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
}

// PyCudaImage_Init
static int PyCudaImage_Init( PyCudaImage* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyCudaImage_Init()\n");
	
	// parse arguments
	int width = 0;
	int height = 0;
	int mapped = 1;
	int freeOnDelete = 1;

	const char* formatStr = "rgb8";
	static char* kwlist[] = {"width", "height", "format", "mapped", "freeOnDelete", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "ii|sii", kwlist, &width, &height, &formatStr, &mapped, &freeOnDelete))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaImage.__init()__ failed to parse args tuple");
		LogDebug(LOG_PY_UTILS "cudaImage.__init()__ failed to parse args tuple\n");
		return -1;
	}
    
	if( width < 0 || height < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaImage.__init()__ had invalid width/height");
		return -1;
	}

	const imageFormat format = imageFormatFromStr(formatStr);

	if( format == IMAGE_UNKNOWN )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaImage.__init()__ had invalid image format");
		return -1;
	}

	// allocate CUDA memory
	const size_t size = imageFormatSize(format, width, height);

	if( mapped > 0 )
	{
		if( !cudaAllocMapped(&self->base.ptr, size) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaImage.__init()__ failed to allocate CUDA mapped memory");
			return -1;
		}
	}
	else
	{
		if( CUDA_FAILED(cudaMalloc(&self->base.ptr, size)) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaImage.__init()__ failed to allocate CUDA memory");
			return -1;
		}
	}
	
	PyCudaImage_Config(self, self->base.ptr, width, height, format, (mapped > 0) ? true : false, (freeOnDelete > 0) ? true : false);
	return 0;
}

// PyCudaImage_ToString
static PyObject* PyCudaImage_ToString( PyCudaImage* self )
{
	char str[1024];

	sprintf(str, 
		   "<cudaImage object>\n"
		   "   -- ptr:      %p\n"
		   "   -- size:     %zu\n"
		   "   -- width:    %u\n"
		   "   -- height:   %u\n"
		   "   -- channels: %u\n"
		   "   -- format:   %s\n"
		   "   -- mapped:   %s\n"
		   "   -- freeOnDelete: %s\n",
		   self->base.ptr, self->base.size, (uint32_t)self->width, (uint32_t)self->height, (uint32_t)self->shape[2],  
		   imageFormatToStr(self->format), self->base.mapped ? "true" : "false", self->base.freeOnDelete ? "true" : "false");

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

	// return the pixel as a tuple
	PyObject* tuple = PyTuple_New(numComponents);
	const imageBaseType baseType = imageFormatBaseType(self->format);

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
	pyCudaImage_Type.tp_base      = &pyCudaMemory_Type;
	pyCudaImage_Type.tp_methods   = NULL;
	pyCudaImage_Type.tp_getset    = pyCudaImage_GetSet;
	pyCudaImage_Type.tp_as_mapping = &pyCudaImage_AsMapping;
	pyCudaImage_Type.tp_new 	     = PyCudaImage_New;
	pyCudaImage_Type.tp_init	     = (initproc)PyCudaImage_Init;
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

	return (PyObject*)mem;
}

// PyCUDA_RegisterImage
PyObject* PyCUDA_RegisterImage( void* gpuPtr, uint32_t width, uint32_t height, imageFormat format, bool mapped, bool freeOnDelete )
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

	PyCudaImage_Config(mem, gpuPtr, width, height, format, mapped, freeOnDelete);
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
void* PyCUDA_GetImage( PyObject* capsule, int* width, int* height, imageFormat* format )
{
	PyCudaImage* img = PyCUDA_GetImage(capsule);
	void* ptr = NULL;

	if( img != NULL )
	{
		ptr = img->base.ptr;
		*width = img->width;
		*height = img->height;
		*format = img->format;
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

	const char* formatStr = NULL;
	static char* kwlist[] = {"size", "width", "height", "format", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|iiis", kwlist, &size, &width, &height, &formatStr))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() failed to parse arguments");
		return NULL;
	}
		
	if( size <= 0 && (width <= 0 || height <= 0) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() requested size/dimensions are negative or zero");
		return NULL;
	}

	const bool isImage = (width > 0) && (height > 0);
	const imageFormat format = imageFormatFromStr(formatStr);

	if( isImage && format == IMAGE_UNKNOWN )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() invalid format string");
		return NULL;
	}

	if( isImage )
		size = imageFormatSize(format, width, height);

	// allocate memory
	void* ptr = NULL;

	if( !cudaMalloc(&ptr, size) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() failed to allocate memory");
		return NULL;
	}

	return isImage ? PyCUDA_RegisterImage(ptr, width, height, format)
				: PyCUDA_RegisterMemory(ptr, size);
}


// PyCUDA_AllocMapped
PyObject* PyCUDA_AllocMapped( PyObject* self, PyObject* args, PyObject* kwds )
{
	int size = 0;
	float width = 0;
	float height = 0;

	const char* formatStr = NULL;
	static char* kwlist[] = {"size", "width", "height", "format", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|iffs", kwlist, &size, &width, &height, &formatStr))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() failed to parse arguments");
		return NULL;
	}

	if( size <= 0 && (width <= 0 || height <= 0) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() requested size/dimensions are negative or zero");
		return NULL;
	}

	const bool isImage = (width > 0) && (height > 0);
	const imageFormat format = imageFormatFromStr(formatStr);

	if( isImage && format == IMAGE_UNKNOWN )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() invalid format string");
		return NULL;
	}

	if( isImage )
		size = imageFormatSize(format, width, height);

	// allocate memory
	void* ptr = NULL;

	if( !cudaAllocMapped(&ptr, size) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaAllocMapped() failed to allocate memory");
		return NULL;
	}

	return isImage ? PyCUDA_RegisterImage(ptr, width, height, format, true)
				: PyCUDA_RegisterMemory(ptr, size, true);
}


// PyCUDA_Memcpy
PyObject* PyCUDA_Memcpy( PyObject* self, PyObject* args, PyObject* kwds )
{
	PyObject* dst_capsule = NULL;
	PyObject* src_capsule = NULL;
	
	static char* kwlist[]  = {"dst", "src", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &dst_capsule, &src_capsule))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMemcpy() failed to parse arguments");
		return NULL;
	}
	
	// check if the args were reversed in the single-arg version
	if( !src_capsule && dst_capsule != NULL )
	{
		src_capsule = dst_capsule;
		dst_capsule = NULL;
	}
	
	// get the src image
	int src_width = 0;
	int src_height = 0;
	imageFormat src_format = IMAGE_UNKNOWN;
	
	void* src_ptr = PyCUDA_GetImage(src_capsule, &src_width, &src_height, &src_format);
	
	if( !src_ptr ) 
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "failed to get CUDA image from src argument");
		return NULL;
	}
	
	// allocate the dst image (if needed)
	void* dst_ptr = NULL;
	bool dst_allocated = false;
	
	if( !dst_capsule )
	{
		if( !cudaAllocMapped(&dst_ptr, src_width, src_height, src_format) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMemcpy() failed to allocate memory");
			return NULL;
		}
		
		dst_capsule = PyCUDA_RegisterImage(dst_ptr, src_width, src_height, src_format, true);
		dst_allocated = true;
	}
	else
	{
		int dst_width = 0;
		int dst_height = 0;
		imageFormat dst_format = IMAGE_UNKNOWN;
		
		dst_ptr = PyCUDA_GetImage(dst_capsule, &dst_width, &dst_height, &dst_format);
		
		if( !dst_ptr ) 
		{
			PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "failed to get CUDA image from dst argument");
			return NULL;
		}
		
		if( src_width != dst_width || src_height != dst_height || src_format != dst_format )
		{
			PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "src/dst images need to have matching dimensions and formats");
			return NULL;
		}
	}
	
	if( CUDA_FAILED(cudaMemcpy(dst_ptr, src_ptr, imageFormatSize(src_format, src_width, src_height), cudaMemcpyDeviceToDevice)) )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_UTILS "cudaMemcpy() failed to copy memory");
		return NULL;
	}
			
	if( dst_allocated )
		return dst_capsule;
	
	Py_RETURN_NONE;
}

	
// PyCUDA_DeviceSynchronize
PyObject* PyCUDA_DeviceSynchronize( PyObject* self )
{
	CUDA(cudaDeviceSynchronize());
	Py_RETURN_NONE;
}


// PyCUDA_AdaptFontSize
PyObject* PyCUDA_AdaptFontSize( PyObject* self, PyObject* args )
{
	int dim = 0;

	if( !PyArg_ParseTuple(args, "i", &dim) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "adaptFontSize() failed to parse size argument");
		return NULL;
	}
		
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
	// parse arguments
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;
	
	static char* kwlist[] = {"input", "output", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &pyInput, &pyOutput))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaConvertColor() failed to parse args");
		return NULL;
	}
	
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
	if( CUDA_FAILED(cudaConvertColor(input->base.ptr, input->format, output->base.ptr, output->format, input->width, input->height)) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaConvertColor() failed");
		return NULL;
	}

	// return void
	Py_RETURN_NONE;
}


// PyCUDA_Resize
PyObject* PyCUDA_Resize( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;
	
	static char* kwlist[] = {"input", "output", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &pyInput, &pyOutput))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaResize() failed to parse args");
		return NULL;
	}
	
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
	if( CUDA_FAILED(cudaResize(input->base.ptr, input->width, input->height, output->base.ptr, output->width, output->height, output->format)) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaResize() failed");
		return NULL;
	}

	// return void
	Py_RETURN_NONE;
}


// PyCUDA_Crop
PyObject* PyCUDA_Crop( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;
	
	float left, top, right, bottom;
	static char* kwlist[] = {"input", "output", "roi", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO(ffff)", kwlist, &pyInput, &pyOutput, &left, &top, &right, &bottom))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaCrop() failed to parse args");
		return NULL;
	}
	
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
	if( CUDA_FAILED(cudaCrop(input->base.ptr, output->base.ptr, make_int4(left, top, right, bottom), input->width, input->height, input->format)) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaCrop() failed");
		return NULL;
	}

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
	
	static char* kwlist[] = {"input", "inputRange", "output", "outputRange", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O(ff)O(ff)", kwlist, &pyInput, &input_min, &input_max, &pyOutput, &output_min, &output_max))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaNormalize() failed to parse args");
		return NULL;
	}
	
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
	if( CUDA_FAILED(cudaNormalize(input->base.ptr, make_float2(input_min, input_max), output->base.ptr, make_float2(output_min, output_max), output->width, output->height, output->format)) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaNormalize() failed");
		return NULL;
	}

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

	static char* kwlist[] = {"input", "output", "x", "y", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO|ff", kwlist, &pyInput, &pyOutput, &x, &y))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaOverlay() failed to parse args");
		return NULL;
	}
	
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
	if( CUDA_FAILED(cudaOverlay(input->base.ptr, input->width, input->height, output->base.ptr, output->width, output->height, output->format, x, y)) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaOverlay() failed");
		return NULL;
	}

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
	
	static char* kwlist[] = {"input", "center", "radius", "color", "output", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O(ff)fO|O", kwlist, &pyInput, &x, &y, &radius, &pyColor, &pyOutput))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaDrawCircle() failed to parse arguments");
		return NULL;
	}
	
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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to parse color tuple");
		return NULL;
	}

	// run the CUDA function
	if( CUDA_FAILED(cudaDrawCircle(input->base.ptr, output->base.ptr, input->width, input->height, input->format, 
							 x, y, radius, color)) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaDrawCircle() failed to render");
		return NULL;
	}

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

	static char* kwlist[] = {"input", "a", "b", "color", "line_width", "output", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O(ff)(ff)O|fO", kwlist, &pyInput, &x1, &y1, &x2, &y2, &pyColor, &line_width, &pyOutput))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaDrawLine() failed to parse arguments");
		return NULL;
	}
	
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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to parse color tuple");
		return NULL;
	}

	// run the CUDA function
	if( CUDA_FAILED(cudaDrawLine(input->base.ptr, output->base.ptr, input->width, input->height, input->format,
						    x1, y1, x2, y2, color, line_width)) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaDrawLine() failed to render");
		return NULL;
	}

	Py_RETURN_NONE;
}


// PyCUDA_DrawRect
PyObject* PyCUDA_DrawRect( PyObject* self, PyObject* args, PyObject* kwds )
{
	// parse arguments
	PyObject* pyInput  = NULL;
	PyObject* pyOutput = NULL;
	PyObject* pyColor  = NULL;
	
	float left = 0.0f;
	float top = 0.0f;
	float right = 0.0f;
	float bottom = 0.0f;

	static char* kwlist[] = {"input", "rect", "color", "output", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O(ffff)O|O", kwlist, &pyInput, &left, &top, &right, &bottom, &pyColor, &pyOutput))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaDrawRect() failed to parse arguments");
		return NULL;
	}
	
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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to parse color tuple");
		return NULL;
	}

	// run the CUDA function
	if( CUDA_FAILED(cudaDrawRect(input->base.ptr, output->base.ptr, input->width, input->height, input->format,
						    left, top, right, bottom, color)) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaDrawRect() failed to render");
		return NULL;
	}

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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "pyFont.__init()__ failed to parse args tuple");
		return -1;
	}
  
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

	int x = 0;
	int y = 0;

	static char* kwlist[] = {"image", "width", "height", "text", "x", "y", "color", "background", "format", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|iisiiOOs", kwlist, &input, &width, &height, &text, &x, &y, &color, &bg, &format_str))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() failed to parse function arguments");
		return NULL;
	}

	if( !text )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() was not passed in a text string");
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
	self->font->OverlayText(ptr, format, width, height, text, x, y, rgba, bg_rgba);

	// return void
	Py_RETURN_NONE;
}

static PyTypeObject pyFont_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyFont_Methods[] = 
{
	{ "OverlayText", (PyCFunction)PyFont_OverlayText, METH_VARARGS|METH_KEYWORDS, "Render the font overlay for a given text string"},
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
	{ "cudaAllocMapped", (PyCFunction)PyCUDA_AllocMapped, METH_VARARGS|METH_KEYWORDS, "Allocate CUDA ZeroCopy mapped memory" },
	{ "cudaMemcpy", (PyCFunction)PyCUDA_Memcpy, METH_VARARGS|METH_KEYWORDS, "Copy src image to dst image (or if dst provided, return a new image with the contents of src)" },
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

