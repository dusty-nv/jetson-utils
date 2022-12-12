/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "PyLogging.h"
#include "logging.h"


// PyLogging container
typedef struct {
    PyObject_HEAD
} PyLogging_Object;



// Error()
static PyObject* PyLogging_Error( PyObject* cls, PyObject* args )
{
	// parse arguments
	const char* str = NULL;

	if( !PyArg_ParseTuple(args, "s", &str) )
		return NULL;

	LogError("%s\n", str);
	
	Py_RETURN_NONE;
}


// Warning()
static PyObject* PyLogging_Warning( PyObject* cls, PyObject* args )
{
	// parse arguments
	const char* str = NULL;

	if( !PyArg_ParseTuple(args, "s", &str) )
		return NULL;

	LogWarning("%s\n", str);
	
	Py_RETURN_NONE;
}


// Success()
static PyObject* PyLogging_Success( PyObject* cls, PyObject* args )
{
	// parse arguments
	const char* str = NULL;

	if( !PyArg_ParseTuple(args, "s", &str) )
		return NULL;

	LogSuccess("%s\n", str);
	
	Py_RETURN_NONE;
}


// Info()
static PyObject* PyLogging_Info( PyObject* cls, PyObject* args )
{
	// parse arguments
	const char* str = NULL;

	if( !PyArg_ParseTuple(args, "s", &str) )
		return NULL;

	LogInfo("%s\n", str);
	
	Py_RETURN_NONE;
}


// Verbose()
static PyObject* PyLogging_Verbose( PyObject* cls, PyObject* args )
{
	// parse arguments
	const char* str = NULL;

	if( !PyArg_ParseTuple(args, "s", &str) )
		return NULL;
	
	LogVerbose("%s\n", str);
	
	Py_RETURN_NONE;
}


// Debug()
static PyObject* PyLogging_Debug( PyObject* cls, PyObject* args )
{
	// parse arguments
	const char* str = NULL;

	if( !PyArg_ParseTuple(args, "s", &str) )
		return NULL;

	LogDebug("%s\n", str);
	
	Py_RETURN_NONE;
}


// GetLevel()
static PyObject* PyLogging_GetLevel( PyObject* cls )
{
	return PYSTRING_FROM_STRING(Log::LevelToStr(Log::GetLevel()));
}


// SetLevel()
static PyObject* PyLogging_SetLevel( PyObject* cls, PyObject* args )
{
	// parse arguments
	const char* level_str = NULL;

	if( !PyArg_ParseTuple(args, "s", &level_str) )
		return NULL;

	Log::SetLevel(Log::LevelFromStr(level_str));
	
	Py_RETURN_NONE;
}



// GetFilename
static PyObject* PyLogging_GetFilename( PyObject* cls )
{
	return PYSTRING_FROM_STRING(Log::GetFilename());
}


// SetFilename()
static PyObject* PyLogging_SetFilename( PyObject* cls, PyObject* args )
{
	// parse arguments
	const char* filename = NULL;

	if( !PyArg_ParseTuple(args, "s", &filename) )
		return NULL;

	Log::SetFile(filename);
	
	Py_RETURN_NONE;
}


// Usage
static PyObject* PyLogging_Usage( PyObject* cls )
{
	return PYSTRING_FROM_STRING(Log::Usage());
}



//-------------------------------------------------------------------------------
static PyTypeObject pyLogging_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyLogging_Methods[] = 
{
	{ "Error", (PyCFunction)PyLogging_Error, METH_VARARGS | METH_CLASS, "Log an error message"},
	{ "Warning", (PyCFunction)PyLogging_Warning, METH_VARARGS | METH_CLASS, "Log a warning message"},
	{ "Success", (PyCFunction)PyLogging_Success, METH_VARARGS | METH_CLASS, "Log a success message"},
	{ "Info", (PyCFunction)PyLogging_Info, METH_VARARGS | METH_CLASS, "Log an info message"},
	{ "Verbose", (PyCFunction)PyLogging_Verbose, METH_VARARGS | METH_CLASS, "Log a verbose message"},
	{ "Debug", (PyCFunction)PyLogging_Debug, METH_VARARGS | METH_CLASS, "Log a debug message"},
	{ "GetLevel", (PyCFunction)PyLogging_GetLevel, METH_NOARGS | METH_CLASS, "Get the current logging level (as a string)"},
	{ "SetLevel", (PyCFunction)PyLogging_SetLevel, METH_VARARGS | METH_CLASS, "Set the current logging level (as a string)"},
	{ "GetFilename", (PyCFunction)PyLogging_GetFilename, METH_NOARGS | METH_CLASS, "Get the current logging level (as a string)"},
	{ "SetFilename", (PyCFunction)PyLogging_SetFilename, METH_VARARGS | METH_CLASS, "Set the current logging level (as a string)"},
	{ "Usage", (PyCFunction)PyLogging_Usage, METH_NOARGS | METH_CLASS, "Get the command-line usage string"},
	{NULL}  /* Sentinel */
};

// Register types
bool PyLogging_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;

	pyLogging_Type.tp_name 	   = PY_UTILS_MODULE_NAME ".Log";
	pyLogging_Type.tp_basicsize = sizeof(PyLogging_Object);
	pyLogging_Type.tp_flags 	   = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyLogging_Type.tp_methods   = pyLogging_Methods;
	pyLogging_Type.tp_doc  	   = "Logging interface";
	 
	if( PyType_Ready(&pyLogging_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "Logging PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyLogging_Type);
    
	if( PyModule_AddObject(module, "Log", (PyObject*)&pyLogging_Type) < 0 )
	{
		LogError(LOG_PY_UTILS "Log PyModule_AddObject('Log') failed\n");
		return false;
	}

	return true;
}

static PyMethodDef pyLogging_Functions[] = 
{
	{NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyLogging_RegisterFunctions()
{
	return pyLogging_Functions;
}


