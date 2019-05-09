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



// Test function
PyObject* PyCUDA_Test( PyObject* self, PyObject* args )
{
	//PyErr_SetString(PyExc_Exception, "PyCUDA -- invalid object instance");
		//return NULL;
		
	printf("PyCUDA -- test\n");
	
	return Py_BuildValue("s", "test string"); //Py_RETURN_NONE;
}


//-------------------------------------------------------------------------------

static PyMethodDef pyCUDA_Functions[] = 
{
	{ "Test", (PyCFunction)PyCUDA_Test, METH_NOARGS, "Test function"},
	{NULL}  /* Sentinel */
};

// Register functions
PyMethodDef* PyCUDA_RegisterFunctions()
{
	return pyCUDA_Functions;
}

// Register types
bool PyCUDA_RegisterTypes( PyObject* module )
{
	if( !module )
		return false;
	
	return true;
}
