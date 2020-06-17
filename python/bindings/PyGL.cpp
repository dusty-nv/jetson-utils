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

#include "PyGL.h"
#include "PyCUDA.h"

#include "glDisplay.h"
#include "logging.h"


// PyDisplay container
typedef struct {
    PyObject_HEAD
    glDisplay* display;
} PyDisplay_Object;


// New
static PyObject* PyDisplay_New( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyDisplay_New()\n");
	
	// allocate a new container
	PyDisplay_Object* self = (PyDisplay_Object*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_UTILS "glDisplay tp_alloc() failed to allocate a new object");
		return NULL;
	}
	
	self->display = NULL;
	return (PyObject*)self;
}


// Init
static int PyDisplay_Init( PyDisplay_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyDisplay_Init()\n");
	
	// parse arguments
	int width = -1;
	int height = -1;
	float bg_color[] = { 0.05f, 0.05f, 0.05f, 1.0f };
	const char* title = glDisplay::DEFAULT_TITLE;
	static char* kwlist[] = {"title", "width", "height", "r", "g", "b", "a", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|siiffff", kwlist, &title, &width, &height, &bg_color[0], &bg_color[1], &bg_color[2], &bg_color[3]))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay.__init()__ failed to parse args tuple");
		return -1;
	}
  
	// create the display object
	glDisplay* display = glDisplay::Create(title, width, height, bg_color[0], bg_color[1], bg_color[2], bg_color[3]);

	if( !display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to create glDisplay device");
		return -1;
	}

	self->display = display;
	return 0;
}


// Deallocate
static void PyDisplay_Dealloc( PyDisplay_Object* self )
{
	LogDebug(LOG_PY_UTILS "PyDisplay_Dealloc()\n");

	// free the display
	if( self->display != NULL )
	{
		delete self->display;
		self->display = NULL;
	}
	
	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}


// BeginRender
static PyObject* PyDisplay_BeginRender( PyDisplay_Object* self, PyObject* args, PyObject* kwds )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	// parse arguments
	int userEvents = 1;
	static char* kwlist[] = {"userEvents", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &userEvents))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay.BeginRender() failed to parse args tuple");
		return NULL;
	}

	self->display->BeginRender( userEvents > 0 ? true : false );
	Py_RETURN_NONE; 
}


// EndRender
static PyObject* PyDisplay_EndRender( PyDisplay_Object* self )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	self->display->EndRender();
	Py_RETURN_NONE; 
}


// Render
static PyObject* PyDisplay_Render( PyDisplay_Object* self, PyObject* args, PyObject* kwds )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	// parse arguments
	PyObject* capsule = NULL;

	float x = 0.0f;
	float y = 0.0f;

	int width  = 0;
	int height = 0;
	int norm   = 1;

	const char* format_str = "rgba32f";
	static char* kwlist[] = {"image", "width", "height", "x", "y", "normalize", "format", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|iiffis", kwlist, &capsule, &width, &height, &x, &y, &norm, &format_str))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay.Render() failed to parse args tuple");
		return NULL;
	}

	// parse format string
	imageFormat format = imageFormatFromStr(format_str);

	// get pointer to image data
	void* ptr = PyCUDA_GetImage(capsule, &width, &height, &format);

	if( !ptr )
		return NULL;

	// render the image
	self->display->RenderImage(ptr, width, height, format, x, y, norm > 0 ? true : false);

	// return void
	Py_RETURN_NONE;
}


// Render
static PyObject* PyDisplay_RenderOnce( PyDisplay_Object* self, PyObject* args, PyObject* kwds )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	// parse arguments
	PyObject* capsule = NULL;

	float x = 5.0f;
	float y = 30.0f;

	int width  = 0;
	int height = 0;
	int norm   = 1;

	const char* format_str = "rgba32f";
	static char* kwlist[] = {"image", "width", "height", "x", "y", "normalize", "format", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|iiffis", kwlist, &capsule, &width, &height, &x, &y, &norm))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay.RenderOnce() failed to parse args tuple");
		return NULL;
	}

	// parse format string
	imageFormat format = imageFormatFromStr(format_str);

	// get pointer to image data
	void* ptr = PyCUDA_GetImage(capsule, &width, &height, &format);

	if( !ptr )
		return NULL;

	// render the image
	self->display->RenderOnce(ptr, width, height, format, x, y, norm > 0 ? true : false);

	// return void
	Py_RETURN_NONE;
}


// SetTitle
static PyObject* PyDisplay_SetTitle( PyDisplay_Object* self, PyObject* args )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	// parse arguments
	const char* title = NULL;

	if( !PyArg_ParseTuple(args, "s", &title) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay.SetTitle() failed to parse args tuple");
		return NULL;
	}

	if( title != NULL )
		self->display->SetTitle(title);

	Py_RETURN_NONE;
}


// SetBackgroundColor
static PyObject* PyDisplay_SetBackgroundColor( PyDisplay_Object* self, PyObject* args, PyObject* kwds )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	// parse arguments
	float bg_color[] = { 0.05f, 0.05f, 0.05f, 1.0f };
	static char* kwlist[] = {"r", "g", "b", "a", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|ffff", kwlist, &bg_color[0], &bg_color[1], &bg_color[2], &bg_color[3]))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay.SetBackgroundColor() failed to parse args tuple");
		return NULL;
	}

	self->display->SetBackgroundColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3]);
	Py_RETURN_NONE;
}


// GetFPS
static PyObject* PyDisplay_GetFPS( PyDisplay_Object* self )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	return PyFloat_FromDouble(self->display->GetFPS());
}


// GetWidth
static PyObject* PyDisplay_GetWidth( PyDisplay_Object* self )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->display->GetWidth());
}


// GetHeight
static PyObject* PyDisplay_GetHeight( PyDisplay_Object* self )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->display->GetHeight());
}


// IsOpen
static PyObject* PyDisplay_IsOpen( PyDisplay_Object* self )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	PY_RETURN_BOOL(self->display->IsOpen());
}


// IsClosed
static PyObject* PyDisplay_IsClosed( PyDisplay_Object* self )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	PY_RETURN_BOOL(self->display->IsClosed());
}


// ProcessEvents
static PyObject* PyDisplay_ProcessEvents( PyDisplay_Object* self )
{
	if( !self || !self->display )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "glDisplay invalid object instance");
		return NULL;
	}

	self->display->ProcessEvents();
	Py_RETURN_NONE; 
}



//-------------------------------------------------------------------------------
static PyTypeObject PyDisplay_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef PyDisplay_Methods[] = 
{
	{ "BeginRender", (PyCFunction)PyDisplay_BeginRender, METH_VARARGS|METH_KEYWORDS, "Clear window and begin rendering a frame"},
	{ "EndRender", (PyCFunction)PyDisplay_EndRender, METH_NOARGS, "Finish rendering and refresh / flip the backbuffer"},
	{ "Render", (PyCFunction)PyDisplay_Render, METH_VARARGS|METH_KEYWORDS, "Render a CUDA float4 image using OpenGL interop"},
	{ "RenderOnce", (PyCFunction)PyDisplay_RenderOnce, METH_VARARGS|METH_KEYWORDS, "Begin the frame, render a CUDA float4 image using OpenGL interop, and then end the frame"},
	{ "GetFPS", (PyCFunction)PyDisplay_GetFPS, METH_NOARGS, "Return the average frame time (in milliseconds)"},
	{ "GetWidth", (PyCFunction)PyDisplay_GetWidth, METH_NOARGS, "Return the width of the window (in pixels)"},
	{ "GetHeight", (PyCFunction)PyDisplay_GetHeight, METH_NOARGS, "Return the height of the window (in pixels)"},
	{ "IsOpen", (PyCFunction)PyDisplay_IsOpen, METH_NOARGS, "Returns true if the window is open"},
	{ "IsClosed", (PyCFunction)PyDisplay_IsClosed, METH_NOARGS, "Returns true if the window has been closed"},
	{ "SetBackgroundColor", (PyCFunction)PyDisplay_SetBackgroundColor, METH_VARARGS|METH_KEYWORDS, "Set the window background color"},
	{ "SetTitle", (PyCFunction)PyDisplay_SetTitle, METH_VARARGS, "Set the window title string"},
	{ "ProcessEvents", (PyCFunction)PyDisplay_ProcessEvents, METH_NOARGS, "Process UI events"},
	{NULL}  /* Sentinel */
};

// Register types
bool PyGL_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;

	PyDisplay_Type.tp_name 	   = PY_UTILS_MODULE_NAME ".glDisplay";
	PyDisplay_Type.tp_basicsize = sizeof(PyDisplay_Object);
	PyDisplay_Type.tp_flags 	   = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	PyDisplay_Type.tp_methods   = PyDisplay_Methods;
	PyDisplay_Type.tp_new 	   = PyDisplay_New;
	PyDisplay_Type.tp_init	   = (initproc)PyDisplay_Init;
	PyDisplay_Type.tp_dealloc   = (destructor)PyDisplay_Dealloc;
	PyDisplay_Type.tp_doc  	   = "OpenGL display window";
	 
	if( PyType_Ready(&PyDisplay_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "glDisplay PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&PyDisplay_Type);
    
	if( PyModule_AddObject(module, "glDisplay", (PyObject*)&PyDisplay_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "glDisplay PyModule_AddObject('glDisplay') failed\n");
		return false;
	}

	return true;
}

static PyMethodDef PyGL_Functions[] = 
{
	{NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyGL_RegisterFunctions()
{
	return PyGL_Functions;
}


