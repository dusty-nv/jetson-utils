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
#include "cudaFont.h"

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
		   "   -- ptr:    %p\n"
		   "   -- size:   %zu\n"
		   "   -- width:  %u\n"
		   "   -- height: %u\n"
		   "   -- format: %s\n"
		   "   -- mapped: %s\n"
		   "   -- freeOnDelete: %s\n",
		   self->base.ptr, self->base.size, (uint32_t)self->width, (uint32_t)self->height, imageFormatToStr(self->format), 
		   self->base.mapped ? "true" : "false", self->base.freeOnDelete ? "true" : "false");

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

// PyCudaImage_GetFormat
static PyObject* PyCudaImage_GetFormat( PyCudaImage* self, void* closure )
{
	return PYSTRING_FROM_STRING(imageFormatToStr(self->format));
}

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
	pyCudaImage_Type.tp_new 	     = PyCudaImage_New;
	pyCudaImage_Type.tp_init	     = (initproc)PyCudaImage_Init;
	pyCudaImage_Type.tp_dealloc	= NULL; /*(destructor)PyCudaMemory_Dealloc*/;
	pyCudaImage_Type.tp_str		= (reprfunc)PyCudaImage_ToString;
	pyCudaImage_Type.tp_doc  	= "CUDA image";
	 
#if PY_MAJOR_VERSION >= 3
	pyCudaImage_Type.tp_as_buffer = &pyCudaImage_AsBuffer;
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
	int width = 0;
	int height = 0;

	const char* formatStr = NULL;
	static char* kwlist[] = {"size", "width", "height", "format", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|iiis", kwlist, &size, &width, &height, &formatStr))
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

	// run the CUDA function
	if( CUDA_FAILED(cudaConvertColor(input->base.ptr, input->format, output->base.ptr, output->format, input->width, input->height)) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaConvertColor() failed");
		return NULL;
	}

	// return void
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

	int x = 0;
	int y = 0;

	static char* kwlist[] = {"image", "text", "x", "y", "color", "background", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "Os|iiOO", kwlist, &input, &text, &x, &y, &color, &bg))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() failed to parse args (note that the width/height args have been removed)");
		return NULL;
	}

	//if( !output )
	//	output = input;

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

	// verify dimensions
	/*if( width <= 0 || height <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() image dimensions are invalid");
		return NULL;
	}*/

	// get pointer to input image data
	PyCudaImage* img = PyCUDA_GetImage(input);

	if( !img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() failed to get input image pointer from first arg (should be cudaImage)");
		return NULL;
	}

	// get pointer to output image data
	/*void* output_img = PyCUDA_GetPointer(output);

	if( !output_img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.Overlay() failed to get output image pointer from PyCapsule container");
		return NULL;
	}*/

	//LogDebug("cudaFont.Overlay(%p, %p, %i, %i, '%s', %i, %i, (%f, %f, %f, %f))\n", input_img, output_img, width, height, text, x, y, rgba.x, rgba.y, rgba.z, rgba.w);

	// render the font overlay
	self->font->OverlayText(img->base.ptr, img->format, img->width, img->height, text, x, y, rgba, bg_rgba);

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

//-------------------------------------------------------------------------------
static PyMethodDef pyCUDA_Functions[] = 
{
	{ "cudaMalloc", (PyCFunction)PyCUDA_Malloc, METH_VARARGS|METH_KEYWORDS, "Allocated CUDA memory on the GPU with cudaMalloc()" },
	{ "cudaAllocMapped", (PyCFunction)PyCUDA_AllocMapped, METH_VARARGS|METH_KEYWORDS, "Allocate CUDA ZeroCopy mapped memory" },
	{ "cudaDeviceSynchronize", (PyCFunction)PyCUDA_DeviceSynchronize, METH_NOARGS, "Wait for the GPU to complete all work" },
	{ "cudaConvertColor", (PyCFunction)PyCUDA_ConvertColor, METH_VARARGS|METH_KEYWORDS, "Perform colorspace conversion on the GPU" },
	{ "adaptFontSize", (PyCFunction)PyCUDA_AdaptFontSize, METH_VARARGS, "Determine an appropriate font size for the given image dimension" },
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

