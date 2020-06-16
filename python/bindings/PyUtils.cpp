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

#include "PyUtils.h"

#include "PyGL.h"
#include "PyCUDA.h"
#include "PyVideo.h"
#include "PyCamera.h"
#include "PyImageIO.h"
#include "PyNumpy.h"

#include "logging.h"


const uint32_t pyUtilsMaxFunctions = 128;
      uint32_t pyUtilsNumFunctions = 0;

static PyMethodDef pyUtilsFunctions[pyUtilsMaxFunctions];


// add functions
void PyUtils_AddFunctions( PyMethodDef* functions )
{
	uint32_t count = 0;
		
	if( !functions )
		return;
	
	while(true)
	{
		if( !functions[count].ml_name || !functions[count].ml_meth )
			break;
		
		if( pyUtilsNumFunctions >= pyUtilsMaxFunctions - 1 )
		{
			LogError(LOG_PY_UTILS "exceeded max number of functions to register (%u)\n", pyUtilsMaxFunctions);
			return;
		}
		
		memcpy(pyUtilsFunctions + pyUtilsNumFunctions, functions + count, sizeof(PyMethodDef));
		
		pyUtilsNumFunctions++;
		count++;
	}
}


// register functions
bool PyUtils_RegisterFunctions()
{
	LogDebug(LOG_PY_UTILS "registering module functions...\n");
	
	// zero the master list of functions, so it end with NULL sentinel
	memset(pyUtilsFunctions, 0, sizeof(PyMethodDef) * pyUtilsMaxFunctions);
	
	// add functions to the master list
	PyUtils_AddFunctions(PyGL_RegisterFunctions());
	PyUtils_AddFunctions(PyCUDA_RegisterFunctions());
	PyUtils_AddFunctions(PyVideo_RegisterFunctions());
	PyUtils_AddFunctions(PyCamera_RegisterFunctions());
	PyUtils_AddFunctions(PyImageIO_RegisterFunctions());
	PyUtils_AddFunctions(PyNumpy_RegisterFunctions());

	LogDebug(LOG_PY_UTILS "done registering module functions\n");
	return true;
}


// register object types
bool PyUtils_RegisterTypes( PyObject* module )
{
	LogDebug(LOG_PY_UTILS "registering module types...\n");
	
	if( !PyGL_RegisterTypes(module) )
		LogError(LOG_PY_UTILS "failed to register OpenGL types\n");

	if( !PyCUDA_RegisterTypes(module) )
		LogError(LOG_PY_UTILS "failed to register CUDA types\n");

	if( !PyVideo_RegisterTypes(module) )
		LogError(LOG_PY_UTILS "failed to register Video types\n");

	if( !PyCamera_RegisterTypes(module) )
		LogError(LOG_PY_UTILS "failed to register Camera types\n");

	if( !PyImageIO_RegisterTypes(module) )
		LogError(LOG_PY_UTILS "failed to register ImageIO types\n");

	if( !PyNumpy_RegisterTypes(module) )
		LogError(LOG_PY_UTILS "failed to register NumPy types\n");

	LogDebug(LOG_PY_UTILS "done registering module types\n");
	return true;
}

#ifdef PYTHON_3
static struct PyModuleDef pyUtilsModuleDef = {
        PyModuleDef_HEAD_INIT,
        "jetson_utils_python",
        NULL,
        -1,
        pyUtilsFunctions
};

PyMODINIT_FUNC
PyInit_jetson_utils_python(void)
{
	LogDebug(LOG_PY_UTILS "initializing Python %i.%i bindings...\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
	
	// register functions
	if( !PyUtils_RegisterFunctions() )
		LogError(LOG_PY_UTILS "failed to register module functions\n");
	
	// create the module
	PyObject* module = PyModule_Create(&pyUtilsModuleDef);
	
	if( !module )
	{
		LogError(LOG_PY_UTILS "PyModule_Create() failed\n");
		return NULL;
	}
	
	// register types
	if( !PyUtils_RegisterTypes(module) )
		LogError(LOG_PY_UTILS "failed to register module types\n");
	
	LogDebug(LOG_PY_UTILS "done Python %i.%i binding initialization\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
	return module;
}

#else
PyMODINIT_FUNC
initjetson_utils_python(void)
{
	LogDebug(LOG_PY_UTILS "initializing Python %i.%i bindings...\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
	
	// register functions
	if( !PyUtils_RegisterFunctions() )
		LogError(LOG_PY_UTILS "failed to register module functions\n");
	
	// create the module
	PyObject* module = Py_InitModule("jetson_utils_python", pyUtilsFunctions);
	
	if( !module )
	{
		LogError(LOG_PY_UTILS "Py_InitModule() failed\n");
		return;
	}
	
	// register types
	if( !PyUtils_RegisterTypes(module) )
		LogError(LOG_PY_UTILS "failed to register module types\n");
	
	LogDebug(LOG_PY_UTILS "done Python %i.%i binding initialization\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
}
#endif


