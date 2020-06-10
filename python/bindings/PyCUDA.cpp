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
		printf(LOG_PY_UTILS "cudaMemory.__init()__ failed to parse args tuple\n");
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

	return (PyObject*)self;
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
		printf(LOG_PY_UTILS "cudaImage.__init()__ failed to parse args tuple\n");
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

	self->width = width;
	self->height = height;
	self->format = format;

	self->base.size = size;
	self->base.mapped = (mapped > 0) ? true : false;
	self->base.freeOnDelete = (freeOnDelete > 0) ? true : false;

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
		   self->base.ptr, self->base.size, self->width, self->height, imageFormatToStr(self->format), 
		   self->base.mapped ? "true" : "false", self->base.freeOnDelete ? "true" : "false");

	return PYSTRING_FROM_STRING(str);
}

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
	pyCudaImage_Type.tp_new 	     = PyCudaImage_New;
	pyCudaImage_Type.tp_init	     = (initproc)PyCudaImage_Init;
	pyCudaImage_Type.tp_dealloc	= NULL; /*(destructor)PyCudaMemory_Dealloc*/;
	pyCudaImage_Type.tp_str		= (reprfunc)PyCudaImage_ToString;
	pyCudaImage_Type.tp_doc  	= "CUDA image";
	 
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
// PyCUDA_FreeMalloc
void PyCUDA_FreeMalloc( PyObject* capsule )
{
	printf(LOG_PY_UTILS "freeing cudaMalloc memory\n");

	void* ptr = PyCapsule_GetPointer(capsule, CUDA_MALLOC_MEMORY_CAPSULE);

	if( !ptr )
	{
		printf(LOG_PY_UTILS "PyCUDA_FreeMalloc() failed to get pointer from PyCapsule container\n");
		return;
	}

	if( CUDA_FAILED(cudaFree(ptr)) )
	{
		printf(LOG_PY_UTILS "failed to free cudaMalloc memory with cudaFree()\n");
		return;
	}
}


// PyCUDA_RegisterMemory
PyObject* PyCUDA_RegisterMemory( void* gpuPtr, size_t size, bool freeOnDelete )
{
	if( !gpuPtr )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "RegisterMemory() was provided NULL memory pointers");
		return NULL;
	}

#if 0
	// create capsule object
	PyObject* capsule = PyCapsule_New(gpuPtr, CUDA_MALLOC_MEMORY_CAPSULE, freeOnDelete ? PyCUDA_FreeMalloc : NULL);

	if( !capsule )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "RegisterMemory() failed to create PyCapsule container");
		
		if( freeOnDelete )
			CUDA(cudaFree(gpuPtr));

		return NULL;
	}

	return capsule;
#endif

	PyCudaMemory* mem = PyObject_New(PyCudaMemory, &pyCudaMemory_Type);

	if( !mem )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "PyCUDA_RegisterMemory() failed to create a new cudaMemory object");
		return NULL;
	}

	mem->ptr = gpuPtr;
	mem->size = size;
	mem->mapped = false;
	mem->freeOnDelete = freeOnDelete;

	return (PyObject*)mem;
}


// PyCUDA_Malloc
PyObject* PyCUDA_Malloc( PyObject* self, PyObject* args )
{
	int size = 0;

	if( !PyArg_ParseTuple(args, "i", &size) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() failed to parse size argument");
		return NULL;
	}
		
	if( size <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() requested size is negative or zero");
		return NULL;
	}

	// allocate memory
	void* gpuPtr = NULL;

	if( !cudaMalloc(&gpuPtr, size) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaMalloc() failed");
		return NULL;
	}

	return PyCUDA_RegisterMemory(gpuPtr, size);
}


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
PyObject* PyCUDA_RegisterMappedMemory( void* gpuPtr, size_t size, bool freeOnDelete )
{
	if( !gpuPtr )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaRegisterMappedMemory() was provided NULL memory pointers");
		return NULL;
	}

#if 0
	// create capsule object
	PyObject* capsule = PyCapsule_New(cpuPtr, CUDA_MAPPED_MEMORY_CAPSULE, freeOnDelete ? PyCUDA_FreeMapped : NULL);

	if( !capsule )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "RegisterMappedMemory() failed to create PyCapsule container");
		
		if( freeOnDelete )
			CUDA(cudaFreeHost(cpuPtr));

		return NULL;
	}
#endif

	PyCudaMemory* mem = PyObject_New(PyCudaMemory, &pyCudaMemory_Type);

	if( !mem )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "PyCUDA_RegisterMemory() failed to create a new cudaMemory object");
		return NULL;
	}

	mem->ptr = gpuPtr;
	mem->size = size;
	mem->mapped = true;
	mem->freeOnDelete = freeOnDelete;

	return (PyObject*)mem;
}


// PyCUDA_RegisterImage
PyObject* PyCUDA_RegisterImage( void* gpuPtr, uint32_t width, uint32_t height, imageFormat format, bool freeOnDelete )
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

	mem->base.ptr = gpuPtr;
	mem->base.size = imageFormatSize(format, width, height);
	mem->base.mapped = false;
	mem->base.freeOnDelete = freeOnDelete;

	mem->width = width;
	mem->height = height;
	mem->format = format;

	return (PyObject*)mem;
}


// PyCUDA_RegisterMappedImage
PyObject* PyCUDA_RegisterMappedImage( void* gpuPtr, uint32_t width, uint32_t height, imageFormat format, bool freeOnDelete )
{
	if( !gpuPtr )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaRegisterMappedImage() was provided NULL memory pointers");
		return NULL;
	}

	PyCudaImage* mem = PyObject_New(PyCudaImage, &pyCudaImage_Type);

	if( !mem )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "PyCUDA_RegisterMappedImage() failed to create a new cudaImage object");
		return NULL;
	}

	mem->base.ptr = gpuPtr;
	mem->base.size = imageFormatSize(format, width, height);
	mem->base.mapped = true;
	mem->base.freeOnDelete = freeOnDelete;

	mem->width = width;
	mem->height = height;
	mem->format = format;

	return (PyObject*)mem;
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

	return PyCUDA_RegisterMappedMemory(gpuPtr, size);
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
	printf(LOG_PY_UTILS "PyFont_New()\n");
	
	// allocate a new container
	PyFont_Object* self = (PyFont_Object*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_UTILS "cudaFont tp_alloc() failed to allocate a new object");
		printf(LOG_PY_UTILS "cudaFont tp_alloc() failed to allocate a new object\n");
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
	printf(LOG_PY_UTILS "PyFont_Init()\n");
	
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
	printf(LOG_PY_UTILS "PyFont_Dealloc()\n");

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

	int width  = 0;
	int height = 0;

	int x = 0;
	int y = 0;

	static char* kwlist[] = {"image", "width", "height", "text", "x", "y", "color", "background", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "Oiis|iiOO", kwlist, &input, &width, &height, &text, &x, &y, &color, &bg))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() failed to parse args tuple");
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
	if( width <= 0 || height <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() image dimensions are invalid");
		return NULL;
	}

	// get pointer to input image data
	void* input_img = PyCUDA_GetPointer(input);

	if( !input_img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.OverlayText() failed to get input image pointer from PyCapsule container");
		return NULL;
	}

	// get pointer to output image data
	/*void* output_img = PyCUDA_GetPointer(output);

	if( !output_img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "cudaFont.Overlay() failed to get output image pointer from PyCapsule container");
		return NULL;
	}*/

	//printf("cudaFont.Overlay(%p, %p, %i, %i, '%s', %i, %i, (%f, %f, %f, %f))\n", input_img, output_img, width, height, text, x, y, rgba.x, rgba.y, rgba.z, rgba.w);

	// render the font overlay
	self->font->OverlayText((float4*)input_img, width, height, text, x, y, rgba, bg_rgba);

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
		printf(LOG_PY_UTILS "cudaFont PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyFont_Type);
    
	if( PyModule_AddObject(module, "cudaFont", (PyObject*)&pyFont_Type) < 0 )
	{
		printf(LOG_PY_UTILS "cudaFont PyModule_AddObject('cudaFont') failed\n");
		return false;
	}

	return true;
}

//-------------------------------------------------------------------------------
static PyMethodDef pyCUDA_Functions[] = 
{
	{ "cudaMalloc", (PyCFunction)PyCUDA_Malloc, METH_VARARGS, "Allocated CUDA memory on the GPU with cudaMalloc()" },
	{ "cudaAllocMapped", (PyCFunction)PyCUDA_AllocMapped, METH_VARARGS, "Allocate CUDA ZeroCopy mapped memory" },
	{ "cudaDeviceSynchronize", (PyCFunction)PyCUDA_DeviceSynchronize, METH_NOARGS, "Wait for the GPU to complete all work" },
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

