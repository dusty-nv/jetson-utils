/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "PyVideo.h"
#include "PyCUDA.h"

#include "videoSource.h"
#include "videoOutput.h"

#include "logging.h"


// object containers
typedef struct {
    PyObject_HEAD
    videoSource* source;
} PyVideoSource_Object;

typedef struct {
    PyObject_HEAD
    videoOutput* output;
} PyVideoOutput_Object;


//-------------------------------------------------------------------------------
// PyVideoSource_New
static PyObject* PyVideoSource_New( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyVideoSource_New()\n");
	
	// allocate a new container
	PyVideoSource_Object* self = (PyVideoSource_Object*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_UTILS "videoSource tp_alloc() failed to allocate a new object");
		LogError(LOG_PY_UTILS "videoSource tp_alloc() failed to allocate a new object\n");
		return NULL;
	}
	
    self->source = NULL;
    return (PyObject*)self;
}

// PyVideoSource_Init
static int PyVideoSource_Init( PyVideoSource_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyVideoSource_Init()\n");
	
	// parse arguments
	const char* URI = NULL;
	PyObject* argList = NULL;
	int positionArg = -1;
	
	static char* kwlist[] = {"uri", "argv", "positionArg", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sOi", kwlist, &URI, &argList, &positionArg))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoSource.__init()__ failed to parse args tuple");
		LogError(LOG_PY_UTILS "videoSource.__init()__ failed to parse args tuple\n");
		return -1;
	}
  
	// parse argument list
	size_t argc = 0;
	char** argv = NULL;

	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		argc = PyList_Size(argList);

		if( argc != 0 )
		{
			argv = (char**)malloc(sizeof(char*) * argc);

			if( !argv )
			{
				PyErr_SetString(PyExc_MemoryError, LOG_PY_UTILS "videoSource.__init()__ failed to malloc memory for argv list");
				return -1;
			}

			for( size_t n=0; n < argc; n++ )
			{
				PyObject* item = PyList_GetItem(argList, n);
				
				if( !PyArg_Parse(item, "s", &argv[n]) )
				{
					PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoSource.__init()__ failed to parse argv list");
					return -1;
				}

				LogDebug(LOG_PY_UTILS "videSource.__init__() argv[%zu] = '%s'\n", n, argv[n]);
			}
		}
	}

	// create the video source
	videoSource* source = NULL;

	if( URI != NULL && strlen(URI) > 0 )
	{
		if( argc > 0 )
			source = videoSource::Create(URI, argc, argv);
		else
			source = videoSource::Create(URI);
	}
	else
	{
		source = videoSource::Create(argc, argv, positionArg);
	}

	if( !source )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to create videoSource device");
		return -1;
	}

	self->source = source;
	return 0;
}

// PyVideoSource_Dealloc
static void PyVideoSource_Dealloc( PyVideoSource_Object* self )
{
	LogDebug(LOG_PY_UTILS "PyVideoSource_Dealloc()\n");

	// free the network
	if( self->source != NULL )
	{
		delete self->source;
		self->source = NULL;
	}
	
	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}

// PyVideoSource_Open
static PyObject* PyVideoSource_Open( PyVideoSource_Object* self )
{
	if( !self || !self->source )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoSource invalid object instance");
		return NULL;
	}

	if( !self->source->Open() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to open videoSource device for streaming");
		return NULL;
	}

	Py_RETURN_NONE; 
}

// PyVideoSource_Close
static PyObject* PyVideoSource_Close( PyVideoSource_Object* self )
{
	if( !self || !self->source )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoSource invalid object instance");
		return NULL;
	}

	self->source->Close();
	Py_RETURN_NONE; 
}

// PyVideoSource_Capture
static PyObject* PyVideoSource_Capture( PyVideoSource_Object* self, PyObject* args, PyObject* kwds )
{
	if( !self || !self->source )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoSource invalid object instance");
		return NULL;
	}

	// parse arguments
	const char* pyFormat = "rgb8";
	int pyTimeout = -1;
	static char* kwlist[] = {"format", "timeout", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|si", kwlist, &pyFormat, &pyTimeout))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoSource.Capture() failed to parse args tuple");
		return NULL;
	}

	// convert signed timeout to unsigned long
	uint64_t timeout = UINT64_MAX;

	if( pyTimeout >= 0 )
		timeout = pyTimeout;

	// convert format string to enum
	const imageFormat format = imageFormatFromStr(pyFormat);
	
	// capture image
	void* ptr = NULL;

	if( !self->source->Capture(&ptr, format, timeout) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoSource failed to capture image");
		return NULL;
	}

	// register memory capsule (videoSource will free the underlying memory when source is deleted)
	return PyCUDA_RegisterImage(ptr, self->source->GetWidth(), self->source->GetHeight(), format, self->source->GetOptions().zeroCopy, false);
}

// PyVideoSource_GetWidth
static PyObject* PyVideoSource_GetWidth( PyVideoSource_Object* self )
{
	if( !self || !self->source )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoSource invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->source->GetWidth());
}

// PyVideoSource_GetHeight
static PyObject* PyVideoSource_GetHeight( PyVideoSource_Object* self )
{
	if( !self || !self->source )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoSource invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->source->GetHeight());
}

// PyVideoSource_GetFrameRate
static PyObject* PyVideoSource_GetFrameRate( PyVideoSource_Object* self )
{
	if( !self || !self->source )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->source->GetFrameRate());
}

// PyVideoSource_IsStreaming
static PyObject* PyVideoSource_IsStreaming( PyVideoSource_Object* self )
{
	if( !self || !self->source )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoSource invalid object instance");
		return NULL;
	}

	PY_RETURN_BOOL(self->source->IsStreaming());
}

// Usage
static PyObject* PyVideoSource_Usage( PyVideoSource_Object* self )
{
	return Py_BuildValue("s", videoSource::Usage());
}

static PyTypeObject pyVideoSource_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyVideoSource_Methods[] = 
{
	{ "Open", (PyCFunction)PyVideoSource_Open, METH_NOARGS, "Open the video source for streaming frames"},
	{ "Close", (PyCFunction)PyVideoSource_Close, METH_NOARGS, "Stop streaming video frames"},
	{ "Capture", (PyCFunction)PyVideoSource_Capture, METH_VARARGS|METH_KEYWORDS, "Capture a frame and return the cudaImage"},
	{ "GetWidth", (PyCFunction)PyVideoSource_GetWidth, METH_NOARGS, "Return the width of the video source (in pixels)"},
	{ "GetHeight", (PyCFunction)PyVideoSource_GetHeight, METH_NOARGS, "Return the height of the video source (in pixels)"},
	{ "GetFrameRate", (PyCFunction)PyVideoSource_GetFrameRate, METH_NOARGS, "Return the frames per second of the video source"},	
	{ "IsStreaming", (PyCFunction)PyVideoSource_IsStreaming, METH_NOARGS, "Return true if the stream is open, return false if closed"},
	{ "Usage", (PyCFunction)PyVideoSource_Usage, METH_NOARGS|METH_STATIC, "Return help text describing the command line options"},		
	{NULL}  /* Sentinel */
};

static bool PyVideoSource_RegisterType( PyObject* module )
{
	pyVideoSource_Type.tp_name 	  = PY_UTILS_MODULE_NAME ".videoSource";
	pyVideoSource_Type.tp_basicsize = sizeof(PyVideoSource_Object);
	pyVideoSource_Type.tp_flags 	  = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyVideoSource_Type.tp_methods   = pyVideoSource_Methods;
	pyVideoSource_Type.tp_new 	  = PyVideoSource_New;
	pyVideoSource_Type.tp_init	  = (initproc)PyVideoSource_Init;
	pyVideoSource_Type.tp_dealloc	  = (destructor)PyVideoSource_Dealloc;
	pyVideoSource_Type.tp_doc  	  = "videoSource interface for cameras, video streams, and images";
	 
	if( PyType_Ready(&pyVideoSource_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "videoSource PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyVideoSource_Type);
    
	if( PyModule_AddObject(module, "videoSource", (PyObject*)&pyVideoSource_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "videoSource PyModule_AddObject('videoSource') failed\n");
		return false;
	}

	return true;
}


//-------------------------------------------------------------------------------
// PyVideoOutput_New
static PyObject* PyVideoOutput_New( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyVideoOutput_New()\n");
	
	// allocate a new container
	PyVideoOutput_Object* self = (PyVideoOutput_Object*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_UTILS "videoOutput tp_alloc() failed to allocate a new object");
		LogError(LOG_PY_UTILS "videoOutput tp_alloc() failed to allocate a new object\n");
		return NULL;
	}
	
    self->output = NULL;
    return (PyObject*)self;
}

// PyVideoOutput_Init
static int PyVideoOutput_Init( PyVideoOutput_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_UTILS "PyVideoOutput_Init()\n");
	
	// parse arguments
	const char* URI = NULL;
	PyObject* argList = NULL;
	int positionArg = -1;
	
	static char* kwlist[] = {"uri", "argv", "positionArg", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sOi", kwlist, &URI, &argList, &positionArg))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput.__init()__ failed to parse args tuple");
		LogError(LOG_PY_UTILS "videoOutput.__init()__ failed to parse args tuple\n");
		return -1;
	}
  
	// parse argument list
	size_t argc = 0;
	char** argv = NULL;
	bool   headless = false;

	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		argc = PyList_Size(argList);

		if( argc != 0 )
		{
			argv = (char**)malloc(sizeof(char*) * argc);

			if( !argv )
			{
				PyErr_SetString(PyExc_MemoryError, LOG_PY_UTILS "videoOutput.__init()__ failed to malloc memory for argv list");
				return -1;
			}

			for( size_t n=0; n < argc; n++ )
			{
				PyObject* item = PyList_GetItem(argList, n);
				
				if( !PyArg_Parse(item, "s", &argv[n]) )
				{
					PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput.__init()__ failed to parse argv list");
					return -1;
				}

				if( strcasecmp(argv[n], "--headless") == 0 )
					headless = true;

				LogDebug(LOG_PY_UTILS "videoSource.__init__() argv[%zu] = '%s'\n", n, argv[n]);
			}
		}
	}

	// create the video source
	videoOutput* source = NULL;

	if( URI != NULL && strlen(URI) > 0 )
		source = videoOutput::Create(URI, argc, argv);
	else
		source = videoOutput::Create(argc, argv, positionArg);

	if( !source )
	{
		//if( headless )
		//{
			LogWarning(LOG_PY_UTILS "no output streams, creating fake null output\n");
			source = videoOutput::CreateNullOutput();
		//}
			
		if( !source )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to create videoOutput device");
			return -1;
		}
	}

	self->output = source;
	return 0;
}

// PyVideoOutput_Dealloc
static void PyVideoOutput_Dealloc( PyVideoOutput_Object* self )
{
	LogDebug(LOG_PY_UTILS "PyVideoOutput_Dealloc()\n");

	// free the network
	if( self->output != NULL )
	{
		delete self->output;
		self->output = NULL;
	}
	
	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}

// PyVideoOutput_Open
static PyObject* PyVideoOutput_Open( PyVideoOutput_Object* self )
{
	if( !self || !self->output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput invalid object instance");
		return NULL;
	}

	if( !self->output->Open() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "failed to open videoOutput device for streaming");
		return NULL;
	}

	Py_RETURN_NONE; 
}

// PyVideoOutput_Close
static PyObject* PyVideoOutput_Close( PyVideoOutput_Object* self )
{
	if( !self || !self->output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput invalid object instance");
		return NULL;
	}

	self->output->Close();
	Py_RETURN_NONE; 
}


// PyVideoOutput_Render
static PyObject* PyVideoOutput_Render( PyVideoOutput_Object* self, PyObject* args, PyObject* kwds )
{
	if( !self || !self->output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput invalid object instance");
		return NULL;
	}

	// parse arguments
	PyObject* capsule = NULL;
	static char* kwlist[] = {"image", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &capsule))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput.Render() failed to parse args tuple");
		return NULL;
	}

	// get pointer to image data
	PyCudaImage* img = PyCUDA_GetImage(capsule);

	if( !img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput.Render() failed to get image pointer from first arg (should be cudaImage)");
		return NULL;
	}

	// render the image
	if( !self->output->Render(img->base.ptr, img->width, img->height, img->format) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput failed to render image");
		return NULL;
	}

	Py_RETURN_NONE;
}

// PyVideoOutput_GetWidth
static PyObject* PyVideoOutput_GetWidth( PyVideoOutput_Object* self )
{
	if( !self || !self->output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->output->GetWidth());
}

// PyVideoOutput_GetHeight
static PyObject* PyVideoOutput_GetHeight( PyVideoOutput_Object* self )
{
	if( !self || !self->output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->output->GetHeight());
}

// PyVideoOutput_GetFrameRate
static PyObject* PyVideoOutput_GetFrameRate( PyVideoOutput_Object* self )
{
	if( !self || !self->output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput invalid object instance");
		return NULL;
	}

	return PyFloat_FromDouble(self->output->GetFrameRate());
}

// PyVideoOutput_IsStreaming
static PyObject* PyVideoOutput_IsStreaming( PyVideoOutput_Object* self )
{
	if( !self || !self->output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput invalid object instance");
		return NULL;
	}

	PY_RETURN_BOOL(self->output->IsStreaming());
}

// PyVideoOutput_SetStatus
static PyObject* PyVideoOutput_SetStatus( PyVideoOutput_Object* self, PyObject* args )
{
	if( !self || !self->output )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput invalid object instance");
		return NULL;
	}

	// parse arguments
	const char* str = NULL;

	if( !PyArg_ParseTuple(args, "s", &str) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_UTILS "videoOutput.SetStatus() failed to parse args tuple");
		return NULL;
	}

	if( str != NULL && strlen(str) > 0 )
		self->output->SetStatus(str);

	Py_RETURN_NONE;
}

// Usage
static PyObject* PyVideoOutput_Usage( PyVideoOutput_Object* self )
{
	return Py_BuildValue("s", videoOutput::Usage());
}

static PyTypeObject pyVideoOutput_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyVideoOutput_Methods[] = 
{
	{ "Open", (PyCFunction)PyVideoOutput_Open, METH_NOARGS, "Open the video output for streaming frames"},
	{ "Close", (PyCFunction)PyVideoOutput_Close, METH_NOARGS, "Stop streaming video frames"},
	{ "Render", (PyCFunction)PyVideoOutput_Render, METH_VARARGS|METH_KEYWORDS, "Render a frame (supplied as a cudaImage)"},
	{ "GetWidth", (PyCFunction)PyVideoOutput_GetWidth, METH_NOARGS, "Return the width of the video output (in pixels)"},
	{ "GetHeight", (PyCFunction)PyVideoOutput_GetHeight, METH_NOARGS, "Return the height of the video output (in pixels)"},
	{ "GetFrameRate", (PyCFunction)PyVideoOutput_GetFrameRate, METH_NOARGS, "Return the frames per second of the video output"},		
	{ "IsStreaming", (PyCFunction)PyVideoOutput_IsStreaming, METH_NOARGS, "Return true if the stream is open, return false if closed"},		
	{ "SetStatus", (PyCFunction)PyVideoOutput_SetStatus, METH_VARARGS, "Set the status string (i.e. window title bar text)"},
     { "Usage", (PyCFunction)PyVideoOutput_Usage, METH_NOARGS|METH_STATIC, "Return help text describing the command line options"},	
	{NULL}  /* Sentinel */
};

static bool PyVideoOutput_RegisterType( PyObject* module )
{
	pyVideoOutput_Type.tp_name 	  = PY_UTILS_MODULE_NAME ".videoOutput";
	pyVideoOutput_Type.tp_basicsize = sizeof(PyVideoOutput_Object);
	pyVideoOutput_Type.tp_flags 	  = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyVideoOutput_Type.tp_methods   = pyVideoOutput_Methods;
	pyVideoOutput_Type.tp_new 	  = PyVideoOutput_New;
	pyVideoOutput_Type.tp_init	  = (initproc)PyVideoOutput_Init;
	pyVideoOutput_Type.tp_dealloc	  = (destructor)PyVideoOutput_Dealloc;
	pyVideoOutput_Type.tp_doc  	  = "videoOutput interface for streaming video and images";
	 
	if( PyType_Ready(&pyVideoOutput_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "videoOutput PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyVideoOutput_Type);
    
	if( PyModule_AddObject(module, "videoOutput", (PyObject*)&pyVideoOutput_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "videoOutput PyModule_AddObject('videoOutput') failed\n");
		return false;
	}

	return true;
}

//-------------------------------------------------------------------------------
// Register types
bool PyVideo_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;

	if( !PyVideoSource_RegisterType(module) )
		return false;

	if( !PyVideoOutput_RegisterType(module) )
		return false;

	return true;
}

static PyMethodDef pyVideo_Functions[] = 
{
	{NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyVideo_RegisterFunctions()
{
	return pyVideo_Functions;
}


