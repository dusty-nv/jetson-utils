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

#include "PyCamera.h"
#include "PyCUDA.h"

#include "gstCamera.h"
#include "logging.h"


// PyCamera container
typedef struct {
    PyObject_HEAD
    gstCamera* camera;
} PyCamera_Object;


// New
static PyObject* PyCamera_New( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyCamera_New()\n");
	
	// allocate a new container
	PyCamera_Object* self = (PyCamera_Object*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_UTILS "gstCamera tp_alloc() failed to allocate a new object");
		LogError(LOG_PY_UTILS "gstCamera tp_alloc() failed to allocate a new object\n");
		return NULL;
	}
	
    self->camera = NULL;
    return (PyObject*)self;
}


// Init
static int PyCamera_Init( PyCamera_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyCamera_Init()\n");
	
	// parse arguments
	int camera_width   = gstCamera::DefaultWidth;
	int camera_height  = gstCamera::DefaultHeight;
	const char* device = NULL;

	static char* kwlist[] = {"width", "height", "camera", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|iis", kwlist, &camera_width, &camera_height, &device))
		return -1;
  
	if( camera_width <= 0 )	
		camera_width = gstCamera::DefaultWidth;

	if( camera_height <= 0 )	
		camera_height = gstCamera::DefaultHeight;

	/*if( camera_width <= 0 || camera_height <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gstCamera.__init__() requested dimensions are out of bounds");
		return NULL;
	}*/

	// create the camera object
	gstCamera* camera = gstCamera::Create(camera_width, camera_height, device);

	if( !camera )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to create gstCamera device");
		return -1;
	}

	self->camera = camera;
	return 0;
}


// Deallocate
static void PyCamera_Dealloc( PyCamera_Object* self )
{
	LogDebug(LOG_PY_UTILS "PyCamera_Dealloc()\n");

	// free the network
	if( self->camera != NULL )
	{
		delete self->camera;
		self->camera = NULL;
	}
	
	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}


// Open
static PyObject* PyCamera_Open( PyCamera_Object* self )
{
	if( !self || !self->camera )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gstCamera invalid object instance");
		return NULL;
	}

	if( !self->camera->Open() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to open gstCamera device for streaming");
		return NULL;
	}

	Py_RETURN_NONE; 
}


// Close
static PyObject* PyCamera_Close( PyCamera_Object* self )
{
	if( !self || !self->camera )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gstCamera invalid object instance");
		return NULL;
	}

	self->camera->Close();
	Py_RETURN_NONE; 
}


// Capture
static PyObject* PyCamera_Capture( PyCamera_Object* self, PyObject* args, PyObject* kwds, const char* default_format="raw" )
{
	if( !self || !self->camera )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gstCamera invalid object instance");
		return NULL;
	}

	// parse arguments
	const char* pyFormat = default_format;
	int pyTimeout  = -1;
	int pyZeroCopy = 1;

	static char* kwlist[] = {"format", "timeout", "zeroCopy", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sii", kwlist, &pyFormat, &pyTimeout, &pyZeroCopy))
		return NULL;

	// convert signed timeout to unsigned long
	uint64_t timeout = UINT64_MAX;

	if( pyTimeout >= 0 )
		timeout = pyTimeout;

	// convert format string to enum
	const imageFormat format = imageFormatFromStr(pyFormat);

	// convert int zeroCopy to boolean
	const bool zeroCopy = pyZeroCopy <= 0 ? false : true;

	// capture image
	void* ptr = NULL;

	self->camera->SetZeroCopy(zeroCopy);  // takes effect on the first call only

	if( !self->camera->Capture(&ptr, format, timeout) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gstCamera failed to Capture()");
		return NULL;
	}

	// register memory capsule (gstCamera will free the underlying memory when camera is deleted)
	PyObject* capsule = PyCUDA_RegisterImage(ptr, self->camera->GetWidth(), self->camera->GetHeight(), format == IMAGE_UNKNOWN ? self->camera->GetRawFormat() : format, self->camera->GetLastTimestamp(), self->camera->GetOptions().zeroCopy, false);

	if( !capsule )
		return NULL;

	// create dimension objects
	PyObject* pyWidth  = PYLONG_FROM_LONG(self->camera->GetWidth());
	PyObject* pyHeight = PYLONG_FROM_LONG(self->camera->GetHeight());

	// return tuple
	PyObject* tuple = PyTuple_Pack(3, capsule, pyWidth, pyHeight);

	Py_DECREF(capsule);
	Py_DECREF(pyWidth);
	Py_DECREF(pyHeight);

	return tuple;
}


static PyObject* PyCamera_CaptureRGBA( PyCamera_Object* self, PyObject* args, PyObject* kwds )
{
	return PyCamera_Capture(self, args, kwds, "rgba32f");
}


//// CaptureRGBA
//static PyObject* PyCamera_CaptureRGBA( PyCamera_Object* self, PyObject* args, PyObject* kwds )
//{
//	if( !self || !self->camera )
//	{
//		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gstCamera invalid object instance");
//		return NULL;
//	}
//
//	// parse arguments
//	int pyTimeout  = -1;
//	int pyZeroCopy = 1;
//
//	static char* kwlist[] = {"timeout", "zeroCopy", NULL};
//
//	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, &pyTimeout, &pyZeroCopy))
//	{
//		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gstCamera.CaptureRGBA() failed to parse args tuple");
//		return NULL;
//	}
//
//	// convert signed timeout to unsigned long
//	uint64_t timeout = UINT64_MAX;
//
//	if( pyTimeout >= 0 )
//		timeout = pyTimeout;
//
//	// convert int zeroCopy to boolean
//	const bool zeroCopy = pyZeroCopy <= 0 ? false : true;
//
//	// capture RGBA
//	float* ptr = NULL;
//
//	if( !self->camera->CaptureRGBA(&ptr, timeout, zeroCopy) )
//	{
//		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gstCamera failed to CaptureRGBA()");
//		return NULL;
//	}
//
//	// register memory capsule (gstCamera will free the underlying memory when camera is deleted)
//	PyObject* capsule = PyCUDA_RegisterImage(ptr, self->camera->GetWidth(), self->camera->GetHeight(), IMAGE_RGBA32F, self->camera->GetLastTimestamp(), zeroCopy, false);
//
//	if( !capsule )
//		return NULL;
//
//	// create dimension objects
//	PyObject* pyWidth  = PYLONG_FROM_LONG(self->camera->GetWidth());
//	PyObject* pyHeight = PYLONG_FROM_LONG(self->camera->GetHeight());
//
//	// return tuple
//	PyObject* tuple = PyTuple_Pack(3, capsule, pyWidth, pyHeight);
//
//	Py_DECREF(capsule);
//	Py_DECREF(pyWidth);
//	Py_DECREF(pyHeight);
//
//	return tuple;
//}


// GetWidth()
static PyObject* PyCamera_GetWidth( PyCamera_Object* self )
{
	if( !self || !self->camera )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gstCamera invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->camera->GetWidth());
}


// GetHeight()
static PyObject* PyCamera_GetHeight( PyCamera_Object* self )
{
	if( !self || !self->camera )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "gstCamera invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->camera->GetHeight());
}



//-------------------------------------------------------------------------------
static PyTypeObject pyCamera_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyCamera_Methods[] = 
{
	{ "Open", (PyCFunction)PyCamera_Open, METH_NOARGS, "Open the camera for streaming frames"},
	{ "Close", (PyCFunction)PyCamera_Close, METH_NOARGS, "Stop streaming camera frames"},
	{ "Capture", (PyCFunction)PyCamera_Capture, METH_VARARGS|METH_KEYWORDS, "Capture a camera frame (in raw format by default)"},
	{ "CaptureRGBA", (PyCFunction)PyCamera_CaptureRGBA, METH_VARARGS|METH_KEYWORDS, "Capture a camera frame and convert it to float4 RGBA"},
	{ "GetWidth", (PyCFunction)PyCamera_GetWidth, METH_NOARGS, "Return the width of the camera (in pixels)"},
	{ "GetHeight", (PyCFunction)PyCamera_GetHeight, METH_NOARGS, "Return the height of the camera (in pixels)"},
	{NULL}  /* Sentinel */
};

// Register types
bool PyCamera_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;

	pyCamera_Type.tp_name 	  = PY_UTILS_MODULE_NAME ".gstCamera";
	pyCamera_Type.tp_basicsize = sizeof(PyCamera_Object);
	pyCamera_Type.tp_flags 	  = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyCamera_Type.tp_methods   = pyCamera_Methods;
	pyCamera_Type.tp_new 	  = PyCamera_New;
	pyCamera_Type.tp_init	  = (initproc)PyCamera_Init;
	pyCamera_Type.tp_dealloc	  = (destructor)PyCamera_Dealloc;
	pyCamera_Type.tp_doc  	  = "MIPI CSI or USB camera using GStreamer";
	 
	if( PyType_Ready(&pyCamera_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "gstCamera PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyCamera_Type);
    
	if( PyModule_AddObject(module, "gstCamera", (PyObject*)&pyCamera_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "gstCamera PyModule_AddObject('gstCamera') failed\n");
		return false;
	}

	return true;
}

static PyMethodDef pyCamera_Functions[] = 
{
	{NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyCamera_RegisterFunctions()
{
	return pyCamera_Functions;
}


